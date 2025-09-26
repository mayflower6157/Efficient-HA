import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.nn import functional as F
import torch
from transformers.cache_utils import (Cache)
import torch.distributed as dist
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
logger = logging.get_logger(__name__)
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
    alpha: float,
    threshold_top_p: float,
    threshold_top_k: int,
    early_exit_layers: List[int],
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
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
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)



    model_forward = self.__call__
    if isinstance(model_kwargs.get("past_key_values"), Cache):
        is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
        if getattr(self, "hf_quantizer", None) is not None:
            is_compileable &= self.hf_quantizer.is_compileable
        is_compileable = is_compileable and not generation_config.disable_compile
        if is_compileable and (
            self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
        ):
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
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
            like_final_outputs = outputs.hidden_states[early_exit_layer]
            logits = lm_head(like_final_outputs)
            logits_dict[early_exit_layer] = logits
        logits_dict[len(outputs.hidden_states)] = lm_head(outputs.hidden_states[-1])
        last_layer_tokens_logits = outputs.logits[:, -1, :]
        last_layer_tokens_probs = nn.functional.softmax(last_layer_tokens_logits, dim=-1).squeeze(dim=0).squeeze(dim=0)
        candidate_tokens_probs, candidate_tokens_ids = torch.topk(last_layer_tokens_probs, dim=-1, k=threshold_top_k)
        candidate_tokens_cumulative_probs = candidate_tokens_probs.cumsum(dim=-1)
        candidate_tokens_indices = torch.searchsorted(candidate_tokens_cumulative_probs, threshold_top_p, right=False)
        candidate_tokens_cutoff_idx = torch.min(candidate_tokens_indices + 1, torch.tensor(threshold_top_k))    
        candidate_tokens_ids = candidate_tokens_ids[:candidate_tokens_cutoff_idx]
            
        stacked_early_exit_layers = torch.stack([logits_dict[i][:, -1, :] for i in early_exit_layers], dim=0)
        softmax_early_exit_layers = F.softmax(stacked_early_exit_layers, dim=-1)
        candidate_tokens_early_exit_probs = softmax_early_exit_layers[:,:,candidate_tokens_ids].squeeze(dim=1) # [10 layers, 10 candidate tokens]
        max_candidate_tokens_idx = torch.argmax(candidate_tokens_early_exit_probs)
        premature_max_probs = candidate_tokens_early_exit_probs.max().item()
        target_layers = max_candidate_tokens_idx // candidate_tokens_early_exit_probs.size(1) 
            
        selected_premature_layer_idx = early_exit_layers[target_layers.item()]
        selected_premature_layer_logits = logits_dict[selected_premature_layer_idx][:, -1, :] # [1, vocab_size]
        indices_to_remove = torch.ones_like(selected_premature_layer_logits)
        indices_to_remove[:, candidate_tokens_ids] = 0
        indices_to_remove = indices_to_remove.bool()
        next_token_logits = outputs.logits[:, -1, :]
        final_token_logits = next_token_logits + alpha * premature_max_probs * selected_premature_layer_logits
        final_token_logits = final_token_logits.masked_fill(indices_to_remove, -float("Inf"))


        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # pre-process distribution
        final_token_scores = logits_processor(input_ids, final_token_logits)
        # final_probs = nn.functional.softmax(final_token_scores, dim=-1)
  

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

        # pre-process distribution


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

        # token selection
        if do_sample:
            probs = nn.functional.softmax(final_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(final_token_scores, dim=-1)

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

def evolve_deco_greedy():

    # sample is now a protected function in the latest Transformers library
    transformers.generation.utils.GenerationMixin._sample = deco_greedy_search
