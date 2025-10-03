import copy
import inspect
import warnings
from loguru import logger
logger.remove()
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.nn import functional as F
import torch
from transformers.cache_utils import (Cache)
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
from transformers.generation.utils import ModelOutput,logging
from transformers.generation.configuration_utils import (

    GenerationConfig,

)
logger_greedy = logging.get_logger(__name__)
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


GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
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
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample
    alpha=generation_config.alpha
    threshold_top_p= generation_config.threshold_top_p
    threshold_top_k= generation_config.threshold_top_k
    early_exit_layers= generation_config.early_exit_layers
    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)


    model_forward = self.__call__
    compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        # If we use FA2 and a static cache, we cannot compile with fullgraph
        if self.config._attn_implementation == "flash_attention_2":
            # only raise warning if the user passed an explicit compile-config
            if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                logger_greedy.warning_once(
                    "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                    "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                )
                generation_config.compile_config.fullgraph = False
        model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True
    lm_head = self.get_output_embeddings()
    if lm_head is None:
        lm_head=self.lm_head
    if lm_head is None:
        raise ValueError("not supported for models that don't have output embeddings.")
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        if is_prefill:
            outputs = self(**model_inputs, return_dict=True)
            is_prefill = False      
        else:
            outputs = model_forward(**model_inputs, return_dict=True)
        logits_dict = {}

        
        for _, early_exit_layer in enumerate(early_exit_layers):
            like_final_outputs = normalize_hidden_state(outputs.hidden_states[early_exit_layer])
            logits = lm_head(like_final_outputs)
            logits_dict[early_exit_layer] = logits

        final_hidden = normalize_hidden_state(outputs.hidden_states[-1])
        final_logits = lm_head(final_hidden)
        logits_dict[len(outputs.hidden_states)] = final_logits
  
        # Build last layer candidate tokens
        last_layer_tokens_logits = outputs.logits[:, -1, :]
        last_layer_tokens_probs = nn.functional.softmax(last_layer_tokens_logits, dim=-1)
        candidate_tokens_probs, candidate_tokens_ids = torch.topk(last_layer_tokens_probs, dim=-1, k=threshold_top_k)

       
        
        # Top-P (nucleus sampling)
        
        '''
        # Original code
        candidate_tokens_cumulative_probs = candidate_tokens_probs.cumsum(dim=-1)
        candidate_tokens_indices = torch.searchsorted(candidate_tokens_cumulative_probs, threshold_top_p, right=False)
        candidate_tokens_cutoff_idx = torch.min(candidate_tokens_indices + 1, torch.tensor(threshold_top_k))    
        candidate_tokens_ids = candidate_tokens_ids[:candidate_tokens_cutoff_idx]
        '''
        '''
        candidate_tokens_ids, candidate_tokens_cutoff_idx = get_top_p_candidates_topk_original(
            candidate_tokens_probs,
            candidate_tokens_ids,
            threshold_top_p=threshold_top_p,
            threshold_top_k=threshold_top_k
        )
        '''
        
         # Fixed code
        candidate_tokens_ids, candidate_tokens_cutoff = get_top_p_candidates_topk_fixed(
            candidate_tokens_probs,
            candidate_tokens_ids,
            threshold_top_p=threshold_top_p,
            threshold_top_k=threshold_top_k
        )
        
        # Normalize IDs → returns safe_ids + mask
        candidate_tokens_ids, candidate_mask = normalize_candidate_tokens_ids(candidate_tokens_ids)
        
                
        '''
        # Early-exit logits Original
        stacked_early_exit_layers = torch.stack([logits_dict[i][:, -1, :] for i in early_exit_layers], dim=0)
        softmax_early_exit_layers = F.softmax(stacked_early_exit_layers, dim=-1)
        candidate_tokens_early_exit_probs = softmax_early_exit_layers[:,:,candidate_tokens_ids].squeeze(dim=1) # [10 layers, 10 candidate tokens]
        '''

        # Early-exit logits
        stacked_early_exit_layers = torch.stack(
            [logits_dict[i][:, -1, :] for i in early_exit_layers], dim=0
        )   # [num_layers, batch, vocab]
        
        softmax_early_exit_layers = F.softmax(stacked_early_exit_layers, dim=-1)  
        # shape: [num_layers, batch, vocab]
        
        # Expand candidate IDs for gather
        # candidate_tokens_ids: [batch, top_k]
        # Ensure candidate IDs are [batch, top_k]
        candidate_tokens_ids = candidate_tokens_ids.view(candidate_tokens_ids.size(0), -1)
        # Expand candidate IDs to match [num_layers, batch, top_k]
        expanded_ids = candidate_tokens_ids.unsqueeze(0).expand(
            softmax_early_exit_layers.size(0),  # num_layers
            candidate_tokens_ids.size(0),       # batch
            candidate_tokens_ids.size(1)        # top_k
        )
        
        # Gather safely
        candidate_tokens_early_exit_probs = torch.gather(
            softmax_early_exit_layers, 
            dim=-1, 
            index=expanded_ids
        )   # [num_layers, batch, top_k]

        # Apply mask: set padded entries to -inf so they never influence max/selection
        expanded_mask = candidate_mask.unsqueeze(0).expand_as(candidate_tokens_early_exit_probs)
        candidate_tokens_early_exit_probs = candidate_tokens_early_exit_probs.masked_fill(~expanded_mask, float('-inf'))

        ''' Original code
        max_candidate_tokens_idx = torch.argmax(candidate_tokens_early_exit_probs)
        premature_max_probs = candidate_tokens_early_exit_probs.max().item()
        '''

        # Instead of flattening with argmax, do it in two steps

        # Pick the best layer (last dim)
        layer_max_probs, _ = candidate_tokens_early_exit_probs.max(dim=-1)   # [num_layers, batch_size]
        premature_max_probs, selected_premature_layer_idx = select_best_layers(
            layer_max_probs,
            early_exit_layers)
        ''' Original Code
        # target_layers = max_candidate_tokens_idx // candidate_tokens_early_exit_probs.size(1) 
        # selected_premature_layer_idx = early_exit_layers[target_layers.item()]
        # selected_premature_layer_logits = logits_dict[selected_premature_layer_idx][:, -1, :] # [1, vocab_size]
        '''

        # Fixed Code
        # Example: stack logits from all selected layers
        selected_premature_layer_logits = torch.stack(
            [logits_dict[idx][:, -1, :] for idx in selected_premature_layer_idx],
            dim=0
        )  # shape: [num_selected_layers, batch, vocab_size]

        # Log min/max/values for sanity
        
        ''' Original Code
        indices_to_remove = torch.ones_like(selected_premature_layer_logits)
        indices_to_remove[:, candidate_tokens_ids] = 0
        indices_to_remove = indices_to_remove.bool()
        next_token_logits = outputs.logits[:, -1, :]
        final_token_logits = next_token_logits + alpha * premature_max_probs * selected_premature_layer_logits
        final_token_logits = final_token_logits.masked_fill(indices_to_remove, -float("Inf"))
        '''
        
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
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # pre-process distribution
        final_token_scores = logits_processor(input_ids, final_token_logits)
        # final_probs = nn.functional.softmax(final_token_scores, dim=-1)

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
            logger.info(f"[DEBUG] candidate_tokens_ids values (first 50): {candidate_tokens_ids.detach().flatten().cpu()[:50].tolist()}")
        
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
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        # pre-process distributionfinal_token_scores


        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # Check tensor health
        if torch.isnan(final_token_scores).any() or torch.isinf(final_token_scores).any():
            # Move to CPU for safe inspection
            final_scores_cpu = final_token_scores.detach().to("cpu")
        
            logger.error(
                f"Invalid final_token_scores detected! "
                f"shape={final_scores_cpu.shape}, dtype={final_scores_cpu.dtype}, "
                f"min={final_scores_cpu.min().item()}, max={final_scores_cpu.max().item()}"
            )
        
            # Optionally log a small sample of values
            logger.error(
                f"Sample values: {final_scores_cpu.flatten()[:50].tolist()}"
            )
            raise ValueError("Invalid final_token_scores detected")
            
        # token selection
        if do_sample:
            probs = nn.functional.softmax(final_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            # ensure probs sums to ~1
            logger.debug(f"probs sum per row: {probs.sum(dim=-1)}")
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(final_token_scores, dim=-1)
        logger.info(f"next_tokens shape: {next_tokens.shape}, values: {next_tokens[:10]}")

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
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

# def deco_greedy_search(
#         self,
#         input_ids: torch.LongTensor,
#         alpha: float,
#         threshold_top_p: float,
#         threshold_top_k: int,
#         early_exit_layers: List[int],
#         logits_processor: Optional[LogitsProcessorList] = None,
#         stopping_criteria: Optional[StoppingCriteriaList] = None,
#         max_length: Optional[int] = None,
#         pad_token_id: Optional[int] = None,
#         eos_token_id: Optional[Union[int, List[int]]] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         output_scores: Optional[bool] = None,
#         return_dict_in_generate: Optional[bool] = None,
#         synced_gpus: bool = False,
#         streamer: Optional["BaseStreamer"] = None,
#         **model_kwargs,
#     ) -> Union[GreedySearchOutput, torch.LongTensor]:
               
#         # init values
#         logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
#         stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
#         if max_length is not None:
#             warnings.warn(
#                 "`max_length` is deprecated in this function, use"
#                 " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
#                 UserWarning,
#             )
#             stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
#         pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
#         eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
#         if isinstance(eos_token_id, int):
#             eos_token_id = [eos_token_id]
#         eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
#         output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
#         output_attentions = (
#             output_attentions if output_attentions is not None else self.generation_config.output_attentions
#         )
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
#         )
#         return_dict_in_generate = (
#             return_dict_in_generate
#             if return_dict_in_generate is not None
#             else self.generation_config.return_dict_in_generate
#         )

#         # init attention / hidden states / scores tuples
#         scores = () if (return_dict_in_generate and output_scores) else None
#         decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
#         cross_attentions = () if (return_dict_in_generate and output_attentions) else None
#         decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

#         # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
#         if return_dict_in_generate and self.config.is_encoder_decoder:
#             encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
#             encoder_hidden_states = (
#                 model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
#             )

#         # keep track of which sequences are already finished
#         unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

#         this_peer_finished = False  # used by synced_gpus only
#         while True:
#             if synced_gpus:
#                 # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
#                 # The following logic allows an early break if all peers finished generating their sequence
#                 this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
#                 # send 0.0 if we finished, 1.0 otherwise
#                 dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
#                 # did all peers finish? the reduced sum will be 0.0 then
#                 if this_peer_finished_flag.item() == 0.0:
#                     break

#             # prepare model inputs
#             model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)


#             dict_outputs, outputs = self(
#                 **model_inputs,
#                 return_dict=True,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 early_exit_layers = early_exit_layers
#             )

#             if synced_gpus and this_peer_finished:
#                 continue  # don't waste resources running the code we don't need

#             last_layer_tokens_logits = outputs.logits[:, -1, :]
#             last_layer_tokens_probs = nn.functional.softmax(last_layer_tokens_logits, dim=-1).squeeze(dim=0).squeeze(dim=0)
#             candidate_tokens_probs, candidate_tokens_ids = torch.topk(last_layer_tokens_probs, dim=-1, k=threshold_top_k)
#             candidate_tokens_cumulative_probs = candidate_tokens_probs.cumsum(dim=-1)
#             candidate_tokens_indices = torch.searchsorted(candidate_tokens_cumulative_probs, threshold_top_p, right=False)
#             candidate_tokens_cutoff_idx = torch.min(candidate_tokens_indices + 1, torch.tensor(threshold_top_k))    
#             candidate_tokens_ids = candidate_tokens_ids[:candidate_tokens_cutoff_idx]
                
#             stacked_early_exit_layers = torch.stack([dict_outputs[i][:, -1, :] for i in early_exit_layers], dim=0)
#             softmax_early_exit_layers = F.softmax(stacked_early_exit_layers, dim=-1)
#             candidate_tokens_early_exit_probs = softmax_early_exit_layers[:,:,candidate_tokens_ids].squeeze(dim=1) # [10 layers, 10 candidate tokens]
#             max_candidate_tokens_idx = torch.argmax(candidate_tokens_early_exit_probs)
#             premature_max_probs = candidate_tokens_early_exit_probs.max().item()
#             target_layers = max_candidate_tokens_idx // candidate_tokens_early_exit_probs.size(1) 
                
#             selected_premature_layer_idx = early_exit_layers[target_layers.item()]
#             selected_premature_layer_logits = dict_outputs[selected_premature_layer_idx][:, -1, :] # [1, vocab_size]
#             indices_to_remove = torch.ones_like(selected_premature_layer_logits)
#             indices_to_remove[:, candidate_tokens_ids] = 0
#             indices_to_remove = indices_to_remove.bool()
#             next_token_logits = outputs.logits[:, -1, :]
#             final_token_logits = next_token_logits + alpha * premature_max_probs * selected_premature_layer_logits
#             final_token_logits = final_token_logits.masked_fill(indices_to_remove, -float("Inf"))


#             # pre-process distribution
#             next_tokens_scores = logits_processor(input_ids, next_token_logits)

#             # pre-process distribution
#             final_token_scores = logits_processor(input_ids, final_token_logits)
#             # final_probs = nn.functional.softmax(final_token_scores, dim=-1)
#             next_tokens = torch.argmax(final_token_scores, dim=-1)
            
#             # Store scores, attentions and hidden_states when required
#             if return_dict_in_generate:
#                 if output_scores:
#                     scores += (next_tokens_scores,)
#                 if output_attentions:
#                     decoder_attentions += (
#                         (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
#                     )
#                     if self.config.is_encoder_decoder:
#                         cross_attentions += (outputs.cross_attentions,)

#                 if output_hidden_states:
#                     decoder_hidden_states += (
#                         (outputs.decoder_hidden_states,)
#                         if self.config.is_encoder_decoder
#                         else (outputs.hidden_states,)
#                     )
                    
#             # finished sentences should have their next token be a padding token
#             if eos_token_id is not None:
#                 if pad_token_id is None:
#                     raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
#                 next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

#             # update generated ids, model inputs, and length for next step
#             input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
#             if streamer is not None:
#                 streamer.put(next_tokens.cpu())
#             model_kwargs = self._update_model_kwargs_for_generation(
#                 outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
#             )

#             # if eos_token was found in one sentence, set sentence to finished
#             if eos_token_id_tensor is not None:
#                 unfinished_sequences = unfinished_sequences.mul(
#                     next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
#                 )

#                 # stop when each sentence is finished
#                 if unfinished_sequences.max() == 0:
#                     this_peer_finished = True

#             # stop if we exceed the maximum length
#             if stopping_criteria(input_ids, scores):
#                 this_peer_finished = True

#             if this_peer_finished and not synced_gpus:
#                 break
        
        
#         if streamer is not None:
#             streamer.end()

#         if return_dict_in_generate:
#             if self.config.is_encoder_decoder:
#                 return GreedySearchEncoderDecoderOutput(
#                     sequences=input_ids,
#                     scores=scores,
#                     encoder_attentions=encoder_attentions,
#                     encoder_hidden_states=encoder_hidden_states,
#                     decoder_attentions=decoder_attentions,
#                     cross_attentions=cross_attentions,
#                     decoder_hidden_states=decoder_hidden_states,
#                 )
#             else:
#                 return GreedySearchDecoderOnlyOutput(
#                     sequences=input_ids,
#                     scores=scores,
#                     attentions=decoder_attentions,
#                     hidden_states=decoder_hidden_states,
#                 )
#         else:
#             return input_ids


# --- Original full-vocab version ---
def get_top_p_candidates_topk_original(candidate_tokens_probs, candidate_tokens_ids, threshold_top_p, threshold_top_k):
    candidate_tokens_cumulative_probs = candidate_tokens_probs.cumsum(dim=-1)
    if candidate_tokens_cumulative_probs.dim() == 2 and candidate_tokens_cumulative_probs.shape[0] == 1:
        candidate_tokens_cumulative_probs = candidate_tokens_cumulative_probs.squeeze(0)
        candidate_tokens_ids = candidate_tokens_ids.squeeze(0)
    candidate_tokens_indices = torch.searchsorted(candidate_tokens_cumulative_probs, threshold_top_p, right=False)
    candidate_tokens_cutoff_idx = torch.min(candidate_tokens_indices + 1, torch.tensor(threshold_top_k))    
    candidate_tokens_ids = candidate_tokens_ids[:candidate_tokens_cutoff_idx]

    return candidate_tokens_ids, candidate_tokens_cutoff_idx
    
# --- Refactored version with candidate_tokens_ids as input ---
def get_top_p_candidates_topk_fixed(candidate_tokens_probs, candidate_tokens_ids, threshold_top_p, threshold_top_k):
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

    # --- Step 1: Sort candidate probs + ids together ---
    sorted_probs, sorted_indices = torch.sort(candidate_tokens_probs, dim=-1, descending=True)
    sorted_ids = torch.gather(candidate_tokens_ids, -1, sorted_indices)

    # --- Step 2: Compute cumulative probabilities ---
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # --- Step 3: Find cutoff idx via Top-P ---
    threshold_top_p_tensor = torch.full(
        (batch_size,), threshold_top_p,
        device=cumulative_probs.device,
        dtype=cumulative_probs.dtype
    ).unsqueeze(-1)
    cutoffs = torch.searchsorted(cumulative_probs, threshold_top_p_tensor, right=False) + 1
    cutoffs = torch.clamp(cutoffs, max=threshold_top_k)

    # --- Step 4: Slice candidates per batch ---
    candidate_tokens_final = []
    for i in range(batch_size):
        candidate_tokens_final.append(sorted_ids[i, : cutoffs[i]])

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
    next_token_logits = outputs.logits[:, -1, :]                  # [B, V]
    B, V = next_token_logits.shape

    # --- Step 2: Aggregate premature-layer contribution to [B, V]
    if selected_premature_layer_logits.dim() == 3:  # [L, B, V]
        sp = selected_premature_layer_logits
        p = premature_max_probs
        if p.dim() == 2:                           # [L, B] -> [L, B, 1]
            p = p.unsqueeze(-1)
            
        elif p.dim() == 1:                          # [L]
            p = p.unsqueeze(1).unsqueeze(2) 
        logger.debug("sp:", sp.shape, "p:", p.shape)
        layer_boost = (p * sp).sum(dim=0)          # -> [B, V]
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

    logger.debug(f"[select_best_layers] premature_max_probs.shape={premature_max_probs.shape}")
    logger.debug(f"[select_best_layers] best_idx.shape={best_idx.shape}, device={best_idx.device}")

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
            candidate_tokens_ids = candidate_tokens_ids.view(candidate_tokens_ids.size(0), -1)

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
    - If shape is [B, 1, S, H], average across batch -> [1, S, H]
    - Otherwise return as is
    """
    if isinstance(h, tuple):  # in case it's wrapped
        h = h[0]
    if hasattr(h, "dim") and h.dim() == 4:
        # Example: [4, 1, 275, 2048] -> [1, 275, 2048]
        h = h.mean(dim=0)
    return h
    

def evolve_deco_greedy():

    # sample is now a protected function in the latest Transformers library
    transformers.generation.utils.GenerationMixin._sample = deco_greedy_search
