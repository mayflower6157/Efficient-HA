# --- Global DECO state guard ---
DECO_INITIALIZED = False
DECO_CONFIG_CACHE = None

import copy
import json
import inspect
import warnings
import traceback
from loguru import logger

logger.remove()
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.nn import functional as F
import torch
from transformers.cache_utils import Cache
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import os
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import ModelOutput, logging
from transformers.generation.configuration_utils import (
    GenerationConfig,
)

logger_greedy = logging.get_logger(__name__)


class SafeLogitsProcessorList(LogitsProcessorList):
    def __call__(self, input_ids, scores):
        if scores.dtype == torch.bfloat16:
            scores = scores.to(torch.float32)
        scores = super().__call__(input_ids, scores)
        scores = torch.nan_to_num(scores, nan=-1e4, neginf=-1e4)
        return scores.to(torch.bfloat16)


@dataclass
class GreedySearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # premature_layer_dist: Optional[Dict[int, int]] = None


@dataclass
class GreedySearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using greedy search. Hidden states and attention
    weights of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the
    encoder_hidden_states attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)


    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
    """

    sequences: torch.LongTensor
    scores: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[tuple[tuple[tuple[torch.FloatTensor]]]] = None


@dataclass
class GenerateEncoderDecoderOutput(ModelOutput):
    """
    Outputs of encoder-decoder generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
    """

    sequences: torch.LongTensor
    scores: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[tuple[tuple[tuple[torch.FloatTensor]]]] = None


GreedySearchOutput = Union[
    GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput
]
GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]

'''
def deco_greedy_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    # Usage fixed code
    logits_processor = SafeLogitsProcessorList(logits_processor)
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(
        hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
    )
    do_sample = generation_config.do_sample
    alpha = generation_config.alpha
    threshold_top_p = generation_config.threshold_top_p
    threshold_top_k = generation_config.threshold_top_k
    early_exit_layers = generation_config.early_exit_layers
    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )
    model_kwargs = self._get_initial_cache_position(
        cur_len, input_ids.device, model_kwargs
    )

    model_forward = self.__call__
    compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        # If we use FA2 and a static cache, we cannot compile with fullgraph
        if self.config._attn_implementation == "flash_attention_2":
            # only raise warning if the user passed an explicit compile-config
            if (
                generation_config.compile_config is not None
                and generation_config.compile_config.fullgraph
            ):
                logger_greedy.warning_once(
                    "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                    "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                )
                generation_config.compile_config.fullgraph = False
        model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(
            input_ids, generation_config, **model_kwargs
        )
        is_prefill = False
    else:
        is_prefill = True
    lm_head = self.get_output_embeddings()
    if lm_head is None:
        lm_head = self.lm_head
    if lm_head is None:
        raise ValueError("not supported for models that don't have output embeddings.")
    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device
    ):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update(
            {"output_attentions": output_attentions} if output_attentions else {}
        )
        model_inputs.update(
            {"output_hidden_states": output_hidden_states}
            if output_hidden_states
            else {}
        )

        if is_prefill:
            outputs = self(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)
        logits_dict = {}

        for _, early_exit_layer in enumerate(early_exit_layers):
            like_final_outputs = normalize_hidden_state(
                outputs.hidden_states[early_exit_layer]
            )
            logits = lm_head(like_final_outputs)
            logits_dict[early_exit_layer] = logits

        final_hidden = normalize_hidden_state(outputs.hidden_states[-1])
        final_logits = lm_head(final_hidden)
        logits_dict[len(outputs.hidden_states)] = final_logits

        # Build last layer candidate tokens
        last_layer_tokens_logits = outputs.logits[:, -1, :]
        last_layer_tokens_probs = nn.functional.softmax(
            last_layer_tokens_logits, dim=-1
        )
        candidate_tokens_probs, candidate_tokens_ids = torch.topk(
            last_layer_tokens_probs, dim=-1, k=threshold_top_k
        )

        # Top-P (nucleus sampling)

        """
        # Original code
        candidate_tokens_cumulative_probs = candidate_tokens_probs.cumsum(dim=-1)
        candidate_tokens_indices = torch.searchsorted(candidate_tokens_cumulative_probs, threshold_top_p, right=False)
        candidate_tokens_cutoff_idx = torch.min(candidate_tokens_indices + 1, torch.tensor(threshold_top_k))    
        candidate_tokens_ids = candidate_tokens_ids[:candidate_tokens_cutoff_idx]
        """
        """
        candidate_tokens_ids, candidate_tokens_cutoff_idx = get_top_p_candidates_topk_original(
            candidate_tokens_probs,
            candidate_tokens_ids,
            threshold_top_p=threshold_top_p,
            threshold_top_k=threshold_top_k
        )
        """

        # Fixed code
        candidate_tokens_ids, candidate_tokens_cutoff = get_top_p_candidates_topk_fixed(
            candidate_tokens_probs,
            candidate_tokens_ids,
            threshold_top_p=threshold_top_p,
            threshold_top_k=threshold_top_k,
        )

        # Normalize IDs → returns safe_ids + mask
        candidate_tokens_ids, candidate_mask = normalize_candidate_tokens_ids(
            candidate_tokens_ids
        )

        """
        # Early-exit logits Original
        stacked_early_exit_layers = torch.stack([logits_dict[i][:, -1, :] for i in early_exit_layers], dim=0)
        softmax_early_exit_layers = F.softmax(stacked_early_exit_layers, dim=-1)
        candidate_tokens_early_exit_probs = softmax_early_exit_layers[:,:,candidate_tokens_ids].squeeze(dim=1) # [10 layers, 10 candidate tokens]
        """

        # Early-exit logits
        stacked_early_exit_layers = torch.stack(
            [logits_dict[i][:, -1, :] for i in early_exit_layers], dim=0
        )  # [num_layers, batch, vocab]

        softmax_early_exit_layers = F.softmax(stacked_early_exit_layers, dim=-1)
        # shape: [num_layers, batch, vocab]

        # Expand candidate IDs for gather
        # candidate_tokens_ids: [batch, top_k]
        # Ensure candidate IDs are [batch, top_k]
        candidate_tokens_ids = candidate_tokens_ids.view(
            candidate_tokens_ids.size(0), -1
        )
        # Expand candidate IDs to match [num_layers, batch, top_k]
        expanded_ids = candidate_tokens_ids.unsqueeze(0).expand(
            softmax_early_exit_layers.size(0),  # num_layers
            candidate_tokens_ids.size(0),  # batch
            candidate_tokens_ids.size(1),  # top_k
        )

        # Gather safely
        candidate_tokens_early_exit_probs = torch.gather(
            softmax_early_exit_layers, dim=-1, index=expanded_ids
        )  # [num_layers, batch, top_k]

        # Apply mask: set padded entries to -inf so they never influence max/selection
        expanded_mask = candidate_mask.unsqueeze(0).expand_as(
            candidate_tokens_early_exit_probs
        )
        candidate_tokens_early_exit_probs = (
            candidate_tokens_early_exit_probs.masked_fill(~expanded_mask, float("-inf"))
        )

        """ Original code
        max_candidate_tokens_idx = torch.argmax(candidate_tokens_early_exit_probs)
        premature_max_probs = candidate_tokens_early_exit_probs.max().item()
        """

        # Instead of flattening with argmax, do it in two steps

        # Pick the best layer (last dim)
        layer_max_probs, _ = candidate_tokens_early_exit_probs.max(
            dim=-1
        )  # [num_layers, batch_size]
        premature_max_probs, selected_premature_layer_idx = select_best_layers(
            layer_max_probs, early_exit_layers
        )
        """ Original Code
        # target_layers = max_candidate_tokens_idx // candidate_tokens_early_exit_probs.size(1) 
        # selected_premature_layer_idx = early_exit_layers[target_layers.item()]
        # selected_premature_layer_logits = logits_dict[selected_premature_layer_idx][:, -1, :] # [1, vocab_size]
        """

        # Fixed Code
        # Example: stack logits from all selected layers
        selected_premature_layer_logits = torch.stack(
            [logits_dict[idx][:, -1, :] for idx in selected_premature_layer_idx], dim=0
        )  # shape: [num_selected_layers, batch, vocab_size]

        # Log min/max/values for sanity

        """ Original Code
        indices_to_remove = torch.ones_like(selected_premature_layer_logits)
        indices_to_remove[:, candidate_tokens_ids] = 0
        indices_to_remove = indices_to_remove.bool()
        next_token_logits = outputs.logits[:, -1, :]
        final_token_logits = next_token_logits + alpha * premature_max_probs * selected_premature_layer_logits
        final_token_logits = final_token_logits.masked_fill(indices_to_remove, -float("Inf"))
        """

        # Fixed code
        final_token_logits = compute_final_logits(
            outputs,
            candidate_tokens_ids,
            selected_premature_layer_logits,
            premature_max_probs,
            alpha,
        )
        next_token_logits = outputs.logits[:, -1, :]
        # Optional: log stats
        logger.debug(
            f"[SAFEGUARD] final_token_logits: shape={final_token_logits.shape}, "
            f"min={final_token_logits.min().item()}, max={final_token_logits.max().item()}, "
            f"any_inf={torch.isinf(final_token_logits).any().item()}, "
            f"any_nan={torch.isnan(final_token_logits).any().item()}"
        )

        # pre-process distribution
        # Check health
        check_tensor_health("next_token_logits", next_token_logits)
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # pre-process distribution
        # Check health
        check_tensor_health("final_token_logits", final_token_logits)
        logger.info(f"[CHECK] Pre-processor logits summary:")
        logger.info(
            f"  shape={final_token_logits.shape}, "
            f"min={final_token_logits.min().item():.4f}, "
            f"max={final_token_logits.max().item():.4f}, "
            f"any_nan={torch.isnan(final_token_logits).any().item()}, "
            f"any_inf={torch.isinf(final_token_logits).any().item()}"
        )
        final_token_scores = logits_processor(input_ids, final_token_logits)
        # final_probs = nn.functional.softmax(final_token_scores, dim=-1)
        # Post-operation sanity check
        logger.info(f"[CHECK] Post-processor scores summary:")
        logger.info(
            f"  shape={final_token_scores.shape}, "
            f"min={final_token_scores.min().item():.4f}, "
            f"max={final_token_scores.max().item():.4f}, "
            f"any_nan={torch.isnan(final_token_scores).any().item()}, "
            f"any_inf={torch.isinf(final_token_scores).any().item()}"
        )

        # If invalid, log sample values for debugging
        if (
            torch.isnan(final_token_scores).any()
            or torch.isinf(final_token_scores).any()
        ):
            sample_vals = final_token_scores.flatten()[:30].tolist()
            logger.error(f"[DETECT] Invalid final_token_scores! Sample: {sample_vals}")

        logger.info(f"final_token_scores shape: {final_token_scores.shape}")

        if candidate_tokens_ids is not None:
            # Log candidate IDs (detached and moved to CPU so they print nicely)
            logger.info(
                f"[DEBUG] candidate_tokens_ids: "
                f"shape={candidate_tokens_ids.shape}, "
                f"device={candidate_tokens_ids.device}, "
                f"dtype={candidate_tokens_ids.dtype}, "
                f"min={candidate_tokens_ids.min().item()}, "
                f"max={candidate_tokens_ids.max().item()}"
            )
            logger.info(
                f"[DEBUG] candidate_tokens_ids values (first 50): {candidate_tokens_ids.detach().flatten().cpu()[:50].tolist()}"
            )

            # Log final token scores
            logger.info(
                f"[DEBUG] final_token_scores: "
                f"shape={final_token_scores.shape}, "
                f"device={final_token_scores.device}, "
                f"dtype={final_token_scores.dtype}, "
                f"last_dim={final_token_scores.size(-1)}"
            )
            # Safety check before using it for gather
            if candidate_tokens_ids.max() >= final_token_scores.size(-1):
                raise ValueError("Invalid candidate ID: exceeds vocab size!")

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].to(
            copy=True, dtype=torch.float32, device=input_ids.device
        )

        # pre-process distributionfinal_token_scores

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

            # Check health
            check_tensor_health("final_token_scores", final_token_scores)

        # token selection
        if do_sample:
            probs = nn.functional.softmax(final_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            # ensure probs sums to ~1
            logger.debug(f"probs sum per row: {probs.sum(dim=-1)}")
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(final_token_scores, dim=-1)
        logger.info(
            f"next_tokens shape: {next_tokens.shape}, values: {next_tokens[:10]}"
        )

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, scores
        )
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids
'''


def get_num_layers(model):
    # 1. Direct field (LLaMA/Qwen style)
    if hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers

    # 2. Gemma-3 style (nested inside text_config)
    elif hasattr(model.config, "text_config") and hasattr(
        model.config.text_config, "num_hidden_layers"
    ):
        return model.config.text_config.num_hidden_layers

    # 3. Decoder layers (OPT, LLaMA, Gemma, etc.)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)

    # 4. Encoder layers (T5, BART)
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        return len(model.encoder.layers)

    raise ValueError("Could not auto-detect number of layers for this model.")


def get_early_exit_layers(model, n):
    num_layers = get_num_layers(model)
    max_layer_index = num_layers  # last hidden state index = num_layers
    early_exit_layers = list(range(max(1, num_layers - (n - 1)), max_layer_index + 1))
    return early_exit_layers


# --- Original full-vocab version ---
def get_top_p_candidates_topk_original(
    candidate_tokens_probs, candidate_tokens_ids, threshold_top_p, threshold_top_k
):
    candidate_tokens_cumulative_probs = candidate_tokens_probs.cumsum(dim=-1)
    if (
        candidate_tokens_cumulative_probs.dim() == 2
        and candidate_tokens_cumulative_probs.shape[0] == 1
    ):
        candidate_tokens_cumulative_probs = candidate_tokens_cumulative_probs.squeeze(0)
        candidate_tokens_ids = candidate_tokens_ids.squeeze(0)
    candidate_tokens_indices = torch.searchsorted(
        candidate_tokens_cumulative_probs, threshold_top_p, right=False
    )
    candidate_tokens_cutoff_idx = torch.min(
        candidate_tokens_indices + 1, torch.tensor(threshold_top_k)
    )
    candidate_tokens_ids = candidate_tokens_ids[:candidate_tokens_cutoff_idx]

    return candidate_tokens_ids, candidate_tokens_cutoff_idx


# --- Refactored version with candidate_tokens_ids as input ---
def get_top_p_candidates_topk_fixed(
    candidate_tokens_probs, candidate_tokens_ids, threshold_top_p, threshold_top_k
):
    """
    Apply Top-P (nucleus) filtering to a set of candidate token IDs.

    Args:
        candidate_tokens_probs: Tensor of shape [batch_size, top_k] (probs of candidate IDs)
        candidate_tokens_ids: Tensor of shape [batch_size, top_k] (token IDs from topk)
        threshold_top_p: float, cumulative probability threshold
        threshold_top_k: int, maximum number of tokens to keep

    Returns:
        candidate_tokens_final: list of tensors of filtered token indices per batch
        cutoffs: tensor of cutoff indices for each batch
    """
    if candidate_tokens_probs.dim() == 1:
        candidate_tokens_probs = candidate_tokens_probs.unsqueeze(0)
        candidate_tokens_ids = candidate_tokens_ids.unsqueeze(0)

    batch_size, top_k = candidate_tokens_probs.shape

    logger.info(
        f"[get_top_p] Start | batch={batch_size}, top_k={top_k}, "
        f"threshold_top_p={threshold_top_p}, threshold_top_k={threshold_top_k}"
    )

    # --- Step 1: Sort candidate probs + ids together ---
    sorted_probs, sorted_indices = torch.sort(
        candidate_tokens_probs, dim=-1, descending=True
    )
    sorted_ids = torch.gather(candidate_tokens_ids, -1, sorted_indices)

    logger.debug(
        f"[get_top_p] sorted_probs shape={sorted_probs.shape}, "
        f"sorted_ids shape={sorted_ids.shape}"
    )

    # --- Step 2: Compute cumulative probabilities ---
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    logger.debug(
        f"[get_top_p] cumulative_probs range: "
        f"min={cumulative_probs.min().item():.4f}, max={cumulative_probs.max().item():.4f}"
    )

    # --- Step 3: Find cutoff idx via Top-P ---
    threshold_top_p_tensor = torch.full(
        (batch_size,),
        threshold_top_p,
        device=cumulative_probs.device,
        dtype=cumulative_probs.dtype,
    ).unsqueeze(-1)
    cutoffs = (
        torch.searchsorted(cumulative_probs, threshold_top_p_tensor, right=False) + 1
    )
    cutoffs = torch.clamp(cutoffs, max=threshold_top_k)
    logger.info(f"[get_top_p] cutoffs={cutoffs.tolist()}")

    # --- Step 4: Slice candidates per batch ---
    candidate_tokens_final = []
    for i in range(batch_size):
        cutoff_i = int(cutoffs[i].item())

        # ⚠️ Sanity checks
        if cutoff_i > top_k:
            logger.warning(
                f"[get_top_p] cutoff_i={cutoff_i} exceeds top_k={top_k}, clamping."
            )
            cutoff_i = top_k
        if cutoff_i <= 0:
            logger.warning(f"[get_top_p] cutoff_i={cutoff_i} ≤ 0, forcing 1.")
            cutoff_i = 1

        try:
            candidate_tokens_final.append(sorted_ids[i, :cutoff_i])
        except Exception as e:
            traceback.print_exc()
            logger.error(
                f"[get_top_p] Failed at batch {i}: "
                f"sorted_ids.shape={sorted_ids.shape}, cutoff_i={cutoff_i}, err={e}",
                exc=True,
            )
            raise

        # Optional: periodic trace
        if i < 3 or i % 20 == 0:
            logger.debug(
                f"[get_top_p] batch {i}: cutoff={cutoff_i}, "
                f"first5_ids={sorted_ids[i, :min(5, cutoff_i)].tolist()}"
            )

    logger.info("[get_top_p] Completed nucleus filtering.")

    return candidate_tokens_final, cutoffs


def compute_final_logits(
    outputs,
    candidate_tokens_ids,
    selected_premature_layer_logits,
    premature_max_probs,
    alpha: float,
):
    """
    Compute masked final logits for decoding.

    Args:
        outputs: model output with .logits [B, seq_len, V]
        candidate_tokens_ids: tensor [B, K] candidate ids
        selected_premature_layer_logits: tensor [L, B, V] or [B, V]
        premature_max_probs: tensor [L, B] or [L, B, 1] or broadcastable
        alpha: float scaling factor

    Returns:
        final_token_logits: [B, V] masked logits (finite values)
    """

    # --- Step 1: Get final-step logits
    next_token_logits = outputs.logits[:, -1, :]  # [B, V]
    B, V = next_token_logits.shape

    # --- Step 2: Aggregate premature-layer contribution to [B, V]
    if selected_premature_layer_logits.dim() == 3:  # [L, B, V]
        sp = selected_premature_layer_logits
        p = premature_max_probs
        if p.dim() == 2:  # [L, B] -> [L, B, 1]
            p = p.unsqueeze(-1)

        elif p.dim() == 1:  # [L]
            p = p.unsqueeze(1).unsqueeze(2)
        logger.debug("sp:", sp.shape, "p:", p.shape)
        layer_boost = (p * sp).sum(dim=0)  # -> [B, V]
    else:
        layer_boost = selected_premature_layer_logits  # already [B, V]

    # --- Step 3: Combine logits
    final_token_logits = next_token_logits + alpha * layer_boost  # [B, V]

    # --- Step 4: Mask out unwanted tokens
    candidate_tokens_ids = candidate_tokens_ids.to(
        device=final_token_logits.device, dtype=torch.long
    ).clamp_(0, V - 1)

    indices_to_remove = torch.ones(
        (B, V), dtype=torch.bool, device=final_token_logits.device
    )
    indices_to_remove.scatter_(dim=1, index=candidate_tokens_ids, value=False)

    # Safety: ensure at least one token is kept per row
    all_masked = indices_to_remove.all(dim=1)
    if all_masked.any():
        top1 = final_token_logits.argmax(dim=1, keepdim=True)  # [B,1]
        indices_to_remove[all_masked] = True
        indices_to_remove.scatter_(1, top1[all_masked], False)

    # --- Step 5: Apply mask with large finite negative (bf16 safe)
    NEG_LARGE = torch.finfo(final_token_logits.dtype).min
    final_token_logits = final_token_logits.masked_fill(indices_to_remove, NEG_LARGE)

    return final_token_logits


def select_best_layers(layer_max_probs, early_exit_layers):
    """
    Selects the best premature exit layers given the per-layer probabilities.

    Args:
        layer_max_probs (torch.Tensor): [num_layers, batch_size]
        early_exit_layers (list[int]): Original layer indices (e.g., [25..34])

    Returns:
        premature_max_probs (torch.Tensor): [batch_size]
        best_idx (torch.Tensor): [batch_size]
    """

    num_layers, batch_size = layer_max_probs.shape
    logger.info(f"[select_best_layers] layer_max_probs.shape={layer_max_probs.shape}")

    # Base is the first early_exit_layer (e.g., 25)
    base = min(early_exit_layers)
    mapped_layers = [i - base for i in early_exit_layers]

    logger.debug(f"[select_best_layers] early_exit_layers={early_exit_layers}")
    logger.debug(f"[select_best_layers] base={base}, mapped_layers={mapped_layers}")

    # Safety check
    if max(mapped_layers) >= num_layers:
        raise IndexError(
            f"After remapping, indices {mapped_layers} are out of range for "
            f"layer_max_probs with {num_layers} rows"
        )

    # Restrict to early exit layers
    restricted = layer_max_probs[mapped_layers, :]  # [len(E), B]
    logger.debug(f"[select_best_layers] restricted.shape={restricted.shape}")

    # Max over restricted layers per batch
    premature_max_probs, best_idx = restricted.max(dim=0)  # [B], [B]

    logger.debug(
        f"[select_best_layers] premature_max_probs.shape={premature_max_probs.shape}"
    )
    logger.debug(
        f"[select_best_layers] best_idx.shape={best_idx.shape}, device={best_idx.device}"
    )

    # Map best_idx (0..len(E)-1) back to original layer indices
    selected_layers = [early_exit_layers[i.item()] for i in best_idx]
    logger.info(f"[select_best_layers] selected premature layers={selected_layers}")

    return premature_max_probs, selected_layers


def normalize_candidate_tokens_ids(candidate_tokens_ids, pad_value=-1):
    # Log initial type
    logger.info(f"[normalize] Input type={type(candidate_tokens_ids)}")

    if isinstance(candidate_tokens_ids, list):
        if all(x.shape == candidate_tokens_ids[0].shape for x in candidate_tokens_ids):
            candidate_tokens_ids = torch.stack(candidate_tokens_ids, dim=0)
        else:
            candidate_tokens_ids = pad_sequence(
                candidate_tokens_ids, batch_first=True, padding_value=pad_value
            )

    if isinstance(candidate_tokens_ids, torch.Tensor):
        if candidate_tokens_ids.dim() == 1:
            candidate_tokens_ids = candidate_tokens_ids.unsqueeze(0)
        elif candidate_tokens_ids.dim() > 2:
            candidate_tokens_ids = candidate_tokens_ids.view(
                candidate_tokens_ids.size(0), -1
            )

    # Create mask (True = valid, False = padding)
    mask = candidate_tokens_ids != pad_value

    # Replace pad_value with 0 so gather won’t crash
    safe_ids = candidate_tokens_ids.clone()
    safe_ids[~mask] = 0

    logger.info(
        f"[normalize] After reshape: shape={safe_ids.shape}, "
        f"device={safe_ids.device}, dtype={safe_ids.dtype}"
    )
    logger.debug(
        f"[normalize] Values: {safe_ids.tolist()[:10]} "
        f"(min={safe_ids.min().item()}, max={safe_ids.max().item()})"
    )

    return safe_ids, mask


def normalize_hidden_state(h):
    """
    Normalize a single hidden_state tensor.
    - If shape is [PT, B, S, H], average across batch -> [B, S, H]
    - Otherwise return as is
    """
    if isinstance(h, tuple):  # in case it's wrapped
        h = h[0]
    if hasattr(h, "dim") and h.dim() == 4:
        # Example: [4, 1, 275, 2048] -> [1, 275, 2048]
        h = h.mean(dim=0)
    return h


def check_tensor_health(name, tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        traceback.print_exc()
        logger.error(
            f"{name} has invalid values! "
            f"min={tensor.min().item()}, max={tensor.max().item()}",
            exc_info=True,
        )
        raise ValueError(f"{name} contains NaN or Inf.")


def evolve_deco_greedy(model=None, args=None):
    """
    Patch Transformers' GenerationMixin._sample with a DECO-aware greedy search.
    Optionally merges dynamic early-exit layers with static DECO config.

    Args:
        model: (optional) model object for computing num_layers dynamically
        args: (optional) argparse.Namespace with early_exit_layers attribute
    """
    global DECO_INITIALIZED, DECO_CONFIG_CACHE

    if DECO_INITIALIZED:
        logger.info(
            "[DECO] evolve_deco_greedy() already initialized — skipping re-patch."
        )
        logger.info("[DECO] DECO greedy decoding remains active globally.")
        return DECO_CONFIG_CACHE

    # Load DECO config from JSON
    config_path = os.path.join(
        "/home/mayflower/Efficient-HA/configs", "deco_config.json"
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"⚠️ DECO config not found at {config_path}")
    with open(config_path, "r") as f:
        # logger.info(f"[evolve_deco_greedy] Loading DECO config from {config_path}")
        deco_config = json.load(f)

    # ---Compute early-exit layers (only once) ---
    # --- Step 1: compute early-exit layers dynamically if model + args provided ---
    dynamic_layers = None
    if model is not None and args is not None and hasattr(args, "early_exit_layers"):
        try:
            dynamic_layers = get_early_exit_layers(model, args.early_exit_layers)
            logger.info(
                f"[DECO] Dynamically computed early_exit_layers: {dynamic_layers}"
            )
        except Exception as e:
            logger.warning(f"[DECO] Could not compute dynamic early_exit_layers: {e}")

    static_layers = deco_config.get("early_exit_layers", [])
    if not static_layers and dynamic_layers:
        deco_config["early_exit_layers"] = dynamic_layers
    elif static_layers and dynamic_layers:
        merged = sorted(list(set(static_layers + dynamic_layers)))
        deco_config["early_exit_layers"] = merged
        logger.info(f"[DECO] Merged early_exit_layers: {merged}")

    # --- Cache configuration for reuse ---
    DECO_CONFIG_CACHE = deco_config
    DECO_INITIALIZED = True

    # 1️⃣ Define inner function that does the actual decoding
    def deco_greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.


        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if

            `model.config.is_encoder_decoder=True`.
        """
        # all your DECO code moves here ↓↓↓
        # init values

        # Usage fixed code
        logits_processor = SafeLogitsProcessorList(logits_processor)
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(
            hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
        )
        do_sample = generation_config.do_sample

        # alpha = generation_config.alpha
        # threshold_top_p = generation_config.threshold_top_p
        # threshold_top_k = generation_config.threshold_top_k
        alpha = deco_config.get("alpha", getattr(generation_config, "alpha", 0.6))
        threshold_top_p = deco_config.get(
            "threshold_top_p", getattr(generation_config, "threshold_top_p", 0.9)
        )
        threshold_top_k = deco_config.get(
            "threshold_top_k", getattr(generation_config, "threshold_top_k", 20)
        )
        early_exit_layers = deco_config.get(
            "early_exit_layers", getattr(generation_config, "early_exit_layers", [])
        )
        logger.info(f"[evolve_deco_greedy] early_exit_layers: {early_exit_layers}")
        entropy_threshold = deco_config.get("entropy_threshold", 1.6)
        margin_threshold = deco_config.get("margin_threshold", 0.35)
        kl_threshold = deco_config.get("kl_threshold", 0.03)
        persistence = deco_config.get("persistence", 2)
        alpha_schedule = deco_config.get("alpha_schedule", {})
        debug = deco_config.get("debug", False)

        # For persistence
        batch_size, _ = input_ids.shape[:2]
        exit_streak = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)

        # early_exit_layers = generation_config.early_exit_layers
        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        model_kwargs = self._get_initial_cache_position(
            cur_len, input_ids.device, model_kwargs
        )

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(
            model_kwargs, generation_config
        )
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            # If we use FA2 and a static cache, we cannot compile with fullgraph
            if self.config._attn_implementation == "flash_attention_2":
                # only raise warning if the user passed an explicit compile-config
                if (
                    generation_config.compile_config is not None
                    and generation_config.compile_config.fullgraph
                ):
                    logger_greedy.warning_once(
                        "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                        "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                    )
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(
                input_ids, generation_config, **model_kwargs
            )
            is_prefill = False
        else:
            is_prefill = True
        lm_head = self.get_output_embeddings()
        if lm_head is None:
            lm_head = self.lm_head
        if lm_head is None:
            raise ValueError(
                "not supported for models that don't have output embeddings."
            )

        # --- Alpha scheduler helper ---
        def get_dynamic_alpha(
            alpha_schedule, alpha, cur_len, max_new_tokens, early_exit_layer=None
        ):
            """
            Handles both token-wise (temporal) and layer-wise alpha scheduling.
            """
            if not alpha_schedule:
                return alpha

            # --- Case 1: Temporal schedule ---
            if alpha_schedule.get("type", "time") == "time":
                start = alpha_schedule.get("start", alpha)
                end = alpha_schedule.get("end", alpha)
                mode = alpha_schedule.get("mode", "linear")

                progress = min(cur_len / max_new_tokens, 1.0)
                if mode == "linear":
                    return start + progress * (end - start)
                elif mode == "cosine":
                    return start + 0.5 * (1 - math.cos(math.pi * progress)) * (
                        end - start
                    )
                else:
                    return alpha

            # --- Case 2: Layer-wise schedule ---
            elif alpha_schedule.get("type") == "layer":
                layer_map = alpha_schedule.get("schedule", {})
                if early_exit_layer is not None:
                    return layer_map.get(str(early_exit_layer), alpha)
                else:
                    return alpha

            return alpha

        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update(
                {"output_attentions": output_attentions} if output_attentions else {}
            )
            model_inputs.update(
                {"output_hidden_states": output_hidden_states}
                if output_hidden_states
                else {}
            )

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)
            logits_dict = {}

            for _, early_exit_layer in enumerate(early_exit_layers):
                like_final_outputs = normalize_hidden_state(
                    outputs.hidden_states[early_exit_layer]
                )
                logits = lm_head(like_final_outputs)
                logits_dict[early_exit_layer] = logits

            final_hidden = normalize_hidden_state(outputs.hidden_states[-1])
            final_logits = lm_head(final_hidden)
            logits_dict[len(outputs.hidden_states)] = final_logits

            # Build last layer candidate tokens
            last_layer_tokens_logits = outputs.logits[:, -1, :]
            last_layer_tokens_probs = nn.functional.softmax(
                last_layer_tokens_logits, dim=-1
            )
            candidate_tokens_probs, candidate_tokens_ids = torch.topk(
                last_layer_tokens_probs, dim=-1, k=threshold_top_k
            )

            # Top-P (nucleus sampling)

            """
            # Original code
            candidate_tokens_cumulative_probs = candidate_tokens_probs.cumsum(dim=-1)
            candidate_tokens_indices = torch.searchsorted(candidate_tokens_cumulative_probs, threshold_top_p, right=False)
            candidate_tokens_cutoff_idx = torch.min(candidate_tokens_indices + 1, torch.tensor(threshold_top_k))    
            candidate_tokens_ids = candidate_tokens_ids[:candidate_tokens_cutoff_idx]
            """
            """
            candidate_tokens_ids, candidate_tokens_cutoff_idx = get_top_p_candidates_topk_original(
                candidate_tokens_probs,
                candidate_tokens_ids,
                threshold_top_p=threshold_top_p,
                threshold_top_k=threshold_top_k
            )
            """

            # Fixed code
            candidate_tokens_ids, candidate_tokens_cutoff = (
                get_top_p_candidates_topk_fixed(
                    candidate_tokens_probs,
                    candidate_tokens_ids,
                    threshold_top_p=threshold_top_p,
                    threshold_top_k=threshold_top_k,
                )
            )

            # Normalize IDs → returns safe_ids + mask
            candidate_tokens_ids, candidate_mask = normalize_candidate_tokens_ids(
                candidate_tokens_ids
            )

            """
            # Early-exit logits Original
            stacked_early_exit_layers = torch.stack([logits_dict[i][:, -1, :] for i in early_exit_layers], dim=0)
            softmax_early_exit_layers = F.softmax(stacked_early_exit_layers, dim=-1)
            candidate_tokens_early_exit_probs = softmax_early_exit_layers[:,:,candidate_tokens_ids].squeeze(dim=1) # [10 layers, 10 candidate tokens]
            """

            # Early-exit logits
            stacked_early_exit_layers = torch.stack(
                [logits_dict[i][:, -1, :] for i in early_exit_layers], dim=0
            )  # [num_layers, batch, vocab]

            softmax_early_exit_layers = F.softmax(
                stacked_early_exit_layers, dim=-1
            )  # shape: [num_layers, batch, vocab]

            # --- DECO Confidence Metrics ---
            with torch.no_grad():
                # Final layer distribution
                final_probs = F.softmax(final_logits, dim=-1)
                entropy = -torch.sum(final_probs * final_probs.log(), dim=-1)  # [batch]

                # Margin (top1 - top2 gap)
                # Take only the last token’s probability distribution
                final_probs_last = final_probs[:, -1, :]  # [batch, vocab_size]
                sorted_probs, _ = torch.sort(final_probs_last, dim=-1, descending=True)
                margin = sorted_probs[:, 0] - sorted_probs[:, 1]

                # KL divergence: last early-exit vs final logits
                early_probs = F.softmax(stacked_early_exit_layers[-1], dim=-1)
                logger.debug(
                    f"[DECO] early_probs: {early_probs.shape}, final_probs: {final_probs.shape}"
                )
                kl_div = (
                    F.kl_div(early_probs.log(), final_probs_last, reduction="batchmean")
                    .unsqueeze(0)
                    .expand(batch_size)
                )
                entropy = entropy[:, -1]
                logger.debug(
                    f"[DECO] entropy: {entropy.shape}, margin: {margin.shape}, kl_div: {kl_div.shape}"
                )
                # Combine metrics
                confidence_mask = (
                    (entropy < entropy_threshold)
                    & (margin > margin_threshold)
                    & (kl_div < kl_threshold)
                )

                # Update persistence counter
                exit_streak = torch.where(
                    confidence_mask, exit_streak + 1, torch.zeros_like(exit_streak)
                )

                # Decide if allowed to early exit
                can_exit = exit_streak >= persistence

            # Expand candidate IDs for gather
            # candidate_tokens_ids: [batch, top_k]
            # Ensure candidate IDs are [batch, top_k]
            candidate_tokens_ids = candidate_tokens_ids.view(
                candidate_tokens_ids.size(0), -1
            )
            # Expand candidate IDs to match [num_layers, batch, top_k]
            expanded_ids = candidate_tokens_ids.unsqueeze(0).expand(
                softmax_early_exit_layers.size(0),  # num_layers
                candidate_tokens_ids.size(0),  # batch
                candidate_tokens_ids.size(1),  # top_k
            )

            # Gather safely
            candidate_tokens_early_exit_probs = torch.gather(
                softmax_early_exit_layers, dim=-1, index=expanded_ids
            )  # [num_layers, batch, top_k]

            # Apply mask: set padded entries to -inf so they never influence max/selection
            expanded_mask = candidate_mask.unsqueeze(0).expand_as(
                candidate_tokens_early_exit_probs
            )
            candidate_tokens_early_exit_probs = (
                candidate_tokens_early_exit_probs.masked_fill(
                    ~expanded_mask, float("-inf")
                )
            )

            """ Original code
            max_candidate_tokens_idx = torch.argmax(candidate_tokens_early_exit_probs)
            premature_max_probs = candidate_tokens_early_exit_probs.max().item()
            """

            # Instead of flattening with argmax, do it in two steps

            # Pick the best layer (last dim)
            layer_max_probs, _ = candidate_tokens_early_exit_probs.max(
                dim=-1
            )  # [num_layers, batch_size]
            premature_max_probs, selected_premature_layer_idx = select_best_layers(
                layer_max_probs, early_exit_layers
            )

            # Gate early-exit layer selection based on confidence
            if not can_exit.all():
                logger.info(
                    "[DECO] Confidence too low — skipping early exit this round."
                )
                selected_premature_layer_idx = [
                    early_exit_layers[-1]
                ]  # fall back to last layer
            """ Original Code
            # target_layers = max_candidate_tokens_idx // candidate_tokens_early_exit_probs.size(1) 
            # selected_premature_layer_idx = early_exit_layers[target_layers.item()]
            # selected_premature_layer_logits = logits_dict[selected_premature_layer_idx][:, -1, :] # [1, vocab_size]
            """

            # Fixed Code
            # Example: stack logits from all selected layers
            selected_premature_layer_logits = torch.stack(
                [logits_dict[idx][:, -1, :] for idx in selected_premature_layer_idx],
                dim=0,
            )  # shape: [num_selected_layers, batch, vocab_size]

            # Log min/max/values for sanity

            """ Original Code
            indices_to_remove = torch.ones_like(selected_premature_layer_logits)
            indices_to_remove[:, candidate_tokens_ids] = 0
            indices_to_remove = indices_to_remove.bool()
            next_token_logits = outputs.logits[:, -1, :]
            final_token_logits = next_token_logits + alpha * premature_max_probs * selected_premature_layer_logits
            final_token_logits = final_token_logits.masked_fill(indices_to_remove, -float("Inf"))
            """

            # Fixed code
            # --- Compute dynamic alpha (temporal or layer-wise) ---
            alpha_dynamic = get_dynamic_alpha(
                alpha_schedule,
                alpha,
                cur_len,
                getattr(args, "max_tokens", 50),
                early_exit_layer=(
                    selected_premature_layer_idx[0]
                    if selected_premature_layer_idx
                    else None
                ),
            )
            if alpha_dynamic != alpha:
                logger.info(
                    f"[DECO] Dynamic alpha updated from {alpha} → {alpha_dynamic}"
                )
            alpha = alpha_dynamic

            final_token_logits = compute_final_logits(
                outputs,
                candidate_tokens_ids,
                selected_premature_layer_logits,
                premature_max_probs,
                alpha,
            )
            next_token_logits = outputs.logits[:, -1, :]
            # Optional: log stats
            logger.debug(
                f"[SAFEGUARD] final_token_logits: shape={final_token_logits.shape}, "
                f"min={final_token_logits.min().item()}, max={final_token_logits.max().item()}, "
                f"any_inf={torch.isinf(final_token_logits).any().item()}, "
                f"any_nan={torch.isnan(final_token_logits).any().item()}"
            )

            # pre-process distribution
            # Check health
            check_tensor_health("next_token_logits", next_token_logits)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # pre-process distribution
            # Check health
            check_tensor_health("final_token_logits", final_token_logits)
            logger.info(f"[CHECK] Pre-processor logits summary:")
            logger.info(
                f"  shape={final_token_logits.shape}, "
                f"min={final_token_logits.min().item():.4f}, "
                f"max={final_token_logits.max().item():.4f}, "
                f"any_nan={torch.isnan(final_token_logits).any().item()}, "
                f"any_inf={torch.isinf(final_token_logits).any().item()}"
            )
            final_token_scores = logits_processor(input_ids, final_token_logits)
            # final_probs = nn.functional.softmax(final_token_scores, dim=-1)
            # Post-operation sanity check
            logger.info(f"[CHECK] Post-processor scores summary:")
            logger.info(
                f"  shape={final_token_scores.shape}, "
                f"min={final_token_scores.min().item():.4f}, "
                f"max={final_token_scores.max().item():.4f}, "
                f"any_nan={torch.isnan(final_token_scores).any().item()}, "
                f"any_inf={torch.isinf(final_token_scores).any().item()}"
            )

            # If invalid, log sample values for debugging
            if (
                torch.isnan(final_token_scores).any()
                or torch.isinf(final_token_scores).any()
            ):
                sample_vals = final_token_scores.flatten()[:30].tolist()
                traceback.print_exc()
                logger.error(
                    f"[DETECT] Invalid final_token_scores! Sample: {sample_vals}",
                    exc_info=True,
                )

            logger.info(f"final_token_scores shape: {final_token_scores.shape}")

            if candidate_tokens_ids is not None:
                # Log candidate IDs (detached and moved to CPU so they print nicely)
                logger.info(
                    f"[DEBUG] candidate_tokens_ids: "
                    f"shape={candidate_tokens_ids.shape}, "
                    f"device={candidate_tokens_ids.device}, "
                    f"dtype={candidate_tokens_ids.dtype}, "
                    f"min={candidate_tokens_ids.min().item()}, "
                    f"max={candidate_tokens_ids.max().item()}"
                )
                logger.info(
                    f"[DEBUG] candidate_tokens_ids values (first 50): {candidate_tokens_ids.detach().flatten().cpu()[:50].tolist()}"
                )

                # Log final token scores
                logger.info(
                    f"[DEBUG] final_token_scores: "
                    f"shape={final_token_scores.shape}, "
                    f"device={final_token_scores.device}, "
                    f"dtype={final_token_scores.dtype}, "
                    f"last_dim={final_token_scores.size(-1)}"
                )
                # Safety check before using it for gather
                if candidate_tokens_ids.max() >= final_token_scores.size(-1):
                    raise ValueError("Invalid candidate ID: exceeds vocab size!")

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(
                copy=True, dtype=torch.float32, device=input_ids.device
            )

            # pre-process distributionfinal_token_scores

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

                # Check health
                check_tensor_health("final_token_scores", final_token_scores)

            # token selection
            if do_sample:
                probs = nn.functional.softmax(final_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                # ensure probs sums to ~1
                logger.debug(f"probs sum per row: {probs.sum(dim=-1)}")
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(final_token_scores, dim=-1)
            logger.info(
                f"next_tokens shape: {next_tokens.shape}, values: {next_tokens[:10]}"
            )

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, scores
            )
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    # 2️⃣ Patch Transformers to use our custom function
    transformers.generation.utils.GenerationMixin._sample = deco_greedy_search
    logger.success("🧩 [DECO] GenerationMixin._sample successfully patched.")

    return deco_config
