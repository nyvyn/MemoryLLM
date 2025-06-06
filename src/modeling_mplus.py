# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers import LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

class MemoryLMOutputWithPastAndCrossAttentions(CausalLMOutputWithCrossAttentions):
    def __init__(
        self,
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
        cross_attentions=None,
        delta_memory=None,
        last_hidden_state=None,
        retriever_weights=None,
        encoder_retriever_weights=None,
        ltm_indices=None,
    ):
        super().__init__(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
            cross_attentions=cross_attentions,
        )
        self.delta_memory = delta_memory
        self.last_hidden_state = last_hidden_state
        self.retriever_weights = retriever_weights
        self.encoder_retriever_weights = encoder_retriever_weights
        self.ltm_indices = ltm_indices

def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaLinearScalingRotaryEmbedding` is deprecated an will be removed in v4.46. Please use "
            "`LlamaRotaryEmbedding`, which now also does linear scaling (simply pass the model config to __init__)."
        )
        kwargs["rope_type"] = "linear"
        super().__init__(*args, **kwargs)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaDynamicNTKScalingRotaryEmbedding` is deprecated an will be removed in v4.46. Please use "
            "`LlamaRotaryEmbedding`, which now also does dynamic ntk scaling (simply pass the model config to "
            "__init__)."
        )
        kwargs["rope_type"] = "dynamic"
        super().__init__(*args, **kwargs)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, prefix_token_length=0):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos[:, :, prefix_token_length:]) + (rotate_half(q) * sin[:, :, prefix_token_length:])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def get_projector(in_dim, out_dim, num_selector_layers, hidden_act):
    if num_selector_layers > 2:
        raise NotImplementedError
    if num_selector_layers == 2:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim * 2, bias=False),
            hidden_act,
            nn.Linear(out_dim * 2, out_dim, bias=False)
        )
    else:
        return nn.Linear(in_dim, out_dim, bias=False)

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.add_selector = self.config.add_selector if hasattr(self.config, "add_selector") else False
        if hasattr(self.config, "selector_layers") and layer_idx not in self.config.selector_layers:
            self.add_selector = False

        if self.add_selector:
        
            self.query_proj = get_projector(self.hidden_size, config.selector_hidden_dim, config.num_selector_layers, ACT2FN[self.config.hidden_act])
            self.key_proj = get_projector(self.hidden_size, config.selector_hidden_dim, config.num_selector_layers, ACT2FN[self.config.hidden_act])

            if hasattr(self.config, "add_encoder_retriever") and self.config.add_encoder_retriever:
                self.encoder_query_proj = get_projector(self.hidden_size, config.selector_hidden_dim, config.num_selector_layers, ACT2FN[self.config.hidden_act])

            self.detach_hidden_state = True if (hasattr(self.config, "detach_hidden_state") and self.config.detach_hidden_state) else False

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        prefix_token_length: Optional[int] = 0,
        output_retriever_weights: Optional[bool] = False,
        apply_retriever_weights: Optional[torch.Tensor] = None,
        apply_with_gradient: Optional[bool] = False,
        ltm_length: Optional[int] = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states[:, prefix_token_length:])
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        if self.add_selector and output_retriever_weights:
            if self.config.map_from_hidden_states:
                if self.detach_hidden_state:
                    queries = self.query_proj(hidden_states[:, prefix_token_length:].detach())
                    keys = self.key_proj(hidden_states[:, 1+ltm_length:prefix_token_length].detach())
                else:
                    queries = self.query_proj(hidden_states[:, prefix_token_length:])
                    keys = self.key_proj(hidden_states[:, 1+ltm_length:prefix_token_length])
            else:
                if self.detach_hidden_state:
                    queries = self.query_proj(query_states.detach())
                    keys = self.key_proj(key_states[:, 1+ltm_length:prefix_token_length].detach())
                else:
                    queries = self.query_proj(query_states)
                    keys = self.key_proj(key_states[:, 1+ltm_length:prefix_token_length])
            retriever_weights = torch.sigmoid(torch.matmul(queries, keys.transpose(1, 2)))

        else:
            retriever_weights = None

        query_states = query_states.view(bsz, q_len - prefix_token_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, prefix_token_length=prefix_token_length)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        else:
            # causal_mask
            causal_mask = torch.cat(
                [
                    torch.ones(q_len - prefix_token_length, prefix_token_length, dtype=torch.bool),
                    torch.ones(q_len - prefix_token_length, q_len - prefix_token_length, dtype=torch.bool).tril(diagonal=0),
                ], dim=1
            ).to(query_states.device)
            attn_weights.masked_fill_(~causal_mask, torch.finfo(attn_weights.dtype).min)

        # add retriever_weights on attn_weights
        if self.add_selector and output_retriever_weights and apply_retriever_weights:
            attn_weights[:, :, :, 1:prefix_token_length] = attn_weights[:, :, :, 1:prefix_token_length] + torch.log(retriever_weights.unsqueeze(1) + 1e-5)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len - prefix_token_length, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len - prefix_token_length, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, retriever_weights.mean(1) if retriever_weights is not None else None


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        prefix_token_length: Optional[int] = 0,
        output_retriever_weights: Optional[bool] = False,
        ltm_length: Optional[int] = 0,
        return_full_retriever_weights: Optional[bool] = False,
        random_retriever_length: Optional[int] = False,
        training: Optional[bool] = False,
        encoder_query_indices: Optional[torch.LongTensor] = None,
        memory_key_indicators: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states[:, prefix_token_length:])
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        retriever_weights = None
        encoder_retriever_weights = None

        if self.add_selector and output_retriever_weights:

            sentence_hidden_states = hidden_states[:, prefix_token_length:]
            memory_hidden_states = hidden_states[:, 1+ltm_length:prefix_token_length]
            
            if random_retriever_length:
                assert not return_full_retriever_weights
                length = torch.randint(1, sentence_hidden_states.size(1), (1,)).item()
                sentence_hidden_states = sentence_hidden_states[:, :length]

            if encoder_query_indices is not None:
                encoder_query_hidden_states = memory_hidden_states[:, torch.where(encoder_query_indices)[0]]

            if self.detach_hidden_state:
                queries = self.query_proj(sentence_hidden_states.detach())
                keys = self.key_proj(memory_hidden_states.detach())
                if encoder_query_indices is not None:
                    encoder_queries = self.encoder_query_proj(encoder_query_hidden_states.detach())
            else:
                queries = self.query_proj(sentence_hidden_states)
                keys = self.key_proj(memory_hidden_states)
                if encoder_query_indices is not None:
                    encoder_queries = self.encoder_query_proj(encoder_query_hidden_states)  
            

            if not return_full_retriever_weights:
                retriever_weights = torch.sigmoid(torch.matmul(queries, keys.transpose(1, 2)))
                retriever_weights = retriever_weights.mean(dim=1)
                if encoder_query_indices is not None:
                    encoder_retriever_weights = torch.sigmoid(torch.matmul(encoder_queries, keys.transpose(1, 2)))
                    encoder_retriever_weights = encoder_retriever_weights.mean(dim=1)
                
            else:
                retriever_weights = torch.sigmoid(torch.matmul(queries, keys.transpose(1, 2)))
                if encoder_query_indices is not None:
                    raise NotImplementedError
            
        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len - prefix_token_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, prefix_token_length=prefix_token_length)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if encoder_attention_mask is not None:
            # to make sure the attention mask is used, we use sdpa here
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            is_causal = False
            encoder_attention_mask = encoder_attention_mask[prefix_token_length:]

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=encoder_attention_mask.to(query_states.device),
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len - prefix_token_length, -1)
            attn_output = self.o_proj(attn_output)
            return attn_output, None, past_key_value, retriever_weights, encoder_retriever_weights

        # if output_retriever_weights and apply_retriever_weights:
        #     repeated_key_states = repeat_kv(key_states, self.num_key_value_groups)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # if self.add_selector and output_retriever_weights and apply_retriever_weights:
        #     # retriever_weights: [bsz, klen]
        #     # value_states: [bsz, 1 + klen + qlen, num_heads, head_dim]
        #     value_states = torch.cat([
        #         value_states[:, :1],
        #         value_states[:, 1:retriever_weights.shape[1]+1] * retriever_weights.unsqueeze(-1).unsqueeze(-1),
        #         value_states[:, retriever_weights.shape[1]+1:],
        #     ], dim=1)
        #     # value_states[:, 1:retriever_weights.shape[1]+1] = value_states[:, 1:retriever_weights.shape[1]+1] * retriever_weights.unsqueeze(-1).unsqueeze(-1)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len - prefix_token_length,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        # if self.add_selector and output_retriever_weights and apply_retriever_weights:
            
        #     if apply_with_gradient:

        #         attn_logits = torch.matmul(query_states.transpose(1, 2), repeated_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        #         # add mask on attn_probs
        #         assert self.is_causal
        #         causal_mask = torch.cat(
        #             [
        #                 torch.ones(q_len - prefix_token_length, prefix_token_length, dtype=torch.bool),
        #                 torch.ones(q_len - prefix_token_length, q_len - prefix_token_length, dtype=torch.bool).tril(diagonal=0),
        #             ], dim=1
        #         ).to(query_states.device)

        #         attn_logits.masked_fill_(~causal_mask, torch.finfo(attn_logits.dtype).min)
        #         attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True).values
        #         attn_logits = attn_logits.exp()

        #         rescale_factor = attn_logits.sum(dim=-1, keepdim=True).transpose(1,2) / torch.cat([
        #             attn_logits[:, :, :, :1],
        #             attn_logits[:, :, :, 1:retriever_weights.shape[1]+1] * retriever_weights.unsqueeze(1).unsqueeze(1),
        #             attn_logits[:, :, :, retriever_weights.shape[1]+1:],
        #         ], dim=3).transpose(1,2).sum(dim=-1, keepdim=True)
            
        #     else:

        #         raise NotImplementedError
            
        #         with torch.no_grad():

        #             # attn_probs: 
        #             attn_logits = torch.matmul(query_states.transpose(1, 2), repeated_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        #             # add mask on attn_probs
        #             assert self.is_causal
        #             causal_mask = torch.cat(
        #                 [
        #                     torch.ones(q_len - prefix_token_length, prefix_token_length, dtype=torch.bool),
        #                     torch.ones(q_len - prefix_token_length, q_len - prefix_token_length, dtype=torch.bool).tril(diagonal=0),
        #                 ], dim=1
        #             ).to(query_states.device)

        #             attn_logits.masked_fill_(~causal_mask, torch.finfo(attn_logits.dtype).min)
        #             attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True).values
        #             attn_logits = attn_logits.exp()

        #             rescale_factor = attn_logits.sum(dim=-1, keepdim=True).transpose(1,2).detach() / torch.cat([
        #                 attn_logits[:, :, :, :1].detach(),
        #                 attn_logits[:, :, :, 1:retriever_weights.shape[1]+1].detach() * retriever_weights.unsqueeze(1).unsqueeze(1),
        #                 attn_logits[:, :, :, retriever_weights.shape[1]+1:].detach(),
        #             ], dim=3).transpose(1,2).sum(dim=-1, keepdim=True)
        #         rescale_factor = rescale_factor.detach()

        #     attn_output = attn_output * rescale_factor

        attn_output = attn_output.reshape(bsz, q_len - prefix_token_length, -1).contiguous()

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, retriever_weights, encoder_retriever_weights

class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        prefix_token_length: Optional[int] = 0,
        debug=False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            # TODO: add "prefix_token_length" to LlamaAttention
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                prefix_token_length=prefix_token_length,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states[:, prefix_token_length:])
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len - prefix_token_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, prefix_token_length=prefix_token_length)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if (causal_mask is None and q_len > 1 and prefix_token_length == 0) else False

        if prefix_token_length > 0:
            causal_mask = torch.cat(
                [
                    torch.ones(q_len - prefix_token_length, prefix_token_length, dtype=torch.bool),
                    torch.ones(q_len - prefix_token_length, q_len - prefix_token_length, dtype=torch.bool).tril(diagonal=0),
                ], dim=1
            ).to(query_states.device)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len - prefix_token_length, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        prefix_token_length: Optional[int] = 0,
        output_retriever_weights: Optional[bool] = False,
        encoder_query_indices: Optional[torch.Tensor] = None,
        debug=False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states[:, prefix_token_length:]

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            prefix_token_length=prefix_token_length,
            output_retriever_weights=output_retriever_weights,
            encoder_query_indices=encoder_query_indices,
            **kwargs,
        )
        if len(attn_outputs) == 3:
            hidden_states, self_attn_weights, present_key_value = attn_outputs
            retriever_weights = None
        else:
            hidden_states, self_attn_weights, present_key_value, retriever_weights = attn_outputs
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        
        if encoder_query_indices is None:
            outputs += (retriever_weights, )
        else:
            outputs += (retriever_weights, None)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # Ensure past_key_values is a Cache object
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The Llama Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MPlus(LlamaForCausalLM):
    def __init__(self, config):
        LlamaForCausalLM.__init__(self, config)
        
        self.config = config
        self.L = config.num_hidden_layers
        self.d = config.hidden_size
        self.num_tokens = config.num_tokens
        self.drop_memory_per_layer = config.drop_memory_per_layer
        self.add_decoder_lora = config.add_decoder_lora

        # LTM configs
        self.num_ltm_blocks = config.num_ltm_blocks
        self.num_blocks = config.num_blocks - self.num_ltm_blocks
        self.update_ltm_mode = config.update_ltm_mode
        self.initial_rf_when_moving_stm_to_ltm = config.initial_rf_when_moving_stm_to_ltm
        self.decay_frequency = config.decay_frequency
        self.update_ltm_frequency = config.update_ltm_frequency
        self.update_step = 0

        self.add_bos_embedding = config.add_bos_embedding
        self.memory = nn.Parameter(torch.randn([self.L, self.num_blocks * self.num_tokens, self.d]))
        print(f"Memory Pool Parameters: {len(self.memory.reshape(-1)) / 1_000_000_000:.4f} B")
        self.memory.requires_grad = False
        self.new_memory_positional_emb = nn.Parameter(torch.zeros([1, 1, self.d]))

        if config.add_bos_embedding:
            self.bos_embedding = nn.Parameter(torch.randn([self.L, 1, self.d]))

        if config.lora_config is not None:

            from peft import get_peft_model, LoraConfig, TaskType

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=config.lora_config['inference_mode'], 
                r=config.lora_config['r'], 
                lora_alpha=config.lora_config['lora_alpha'], 
                lora_dropout=config.lora_config['lora_dropout'],
                target_modules=config.lora_config.get('target_modules', None)
            )

            get_peft_model(self.model, peft_config)
            if config.add_decoder_lora:
                get_peft_model(self.model, peft_config, adapter_name="decoder_adapter")

        # LTM parameters
        self.ltm = nn.ParameterList([nn.Parameter(torch.randn([config.ltm_initial_size, self.d])) for _ in range(self.L)])
        self.ltm_keys = nn.ParameterList([nn.Parameter(torch.randn([config.ltm_initial_size, config.selector_hidden_dim])) for _ in range(self.L)])
        self.ltm_recall_frequencies = nn.ParameterList([nn.Parameter(torch.zeros([config.ltm_initial_size])) for _ in range(self.L)])
        self.ltm_ages = nn.ParameterList([nn.Parameter(torch.zeros([config.ltm_initial_size])) for _ in range(self.L)])
        self.memory_ages = [np.zeros([self.num_blocks * self.num_tokens]) for _ in range(self.L)]
        self.put_cached_dropped_memory_on_cpu = True

        self.cached_dropped_memories, self.cached_dropped_memory_ages = None, None
        self.cached_dropped_keys = None

        self.initialized = 1

    def put_ltm_to_numpy(self):

        ltm_recall_frequencies = [ltm_rf.detach().float().cpu().numpy() for ltm_rf in self.ltm_recall_frequencies]
        ltm_ages = [ltm_age.detach().int().cpu().numpy() for ltm_age in self.ltm_ages]
        ltm_keys = [ltm_key.data.detach().cpu() for ltm_key in self.ltm_keys]
        ltm = [ltm.data.detach().cpu() for ltm in self.ltm]

        del self.ltm_recall_frequencies
        del self.ltm_ages
        del self.ltm_keys
        del self.ltm

        self.ltm_recall_frequencies = ltm_recall_frequencies
        self.ltm_ages = ltm_ages
        self.ltm_keys = ltm_keys
        self.ltm = ltm

    def merge_cached_memory(self):
        self.update_ltm(self.cached_dropped_memories,
                        self.cached_dropped_memory_ages,
                        device=self.memory.device if self.put_cached_dropped_memory_on_cpu else None,
                        cached_dropped_keys=self.cached_dropped_keys)
        self.update_step = 0
        self.cached_dropped_memories, self.cached_dropped_memory_ages = None, None

    def save_memory(self, path: str):
        """Save memory related tensors and arrays to ``path``."""

        def _to_cpu(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu()
            return x

        state = {
            "memory": _to_cpu(self.memory.data),
            "ltm": [_to_cpu(t) for t in self.ltm],
            "ltm_keys": [_to_cpu(t) for t in self.ltm_keys],
            "ltm_recall_frequencies": [_to_cpu(t) for t in self.ltm_recall_frequencies],
            "ltm_ages": [_to_cpu(t) for t in self.ltm_ages],
            "memory_ages": self.memory_ages,
            "cached_dropped_memories": _to_cpu(self.cached_dropped_memories) if self.cached_dropped_memories is not None else None,
            "cached_dropped_memory_ages": self.cached_dropped_memory_ages,
            "cached_dropped_keys": _to_cpu(self.cached_dropped_keys) if self.cached_dropped_keys is not None else None,
            "put_cached_dropped_memory_on_cpu": self.put_cached_dropped_memory_on_cpu,
            "update_step": self.update_step,
        }

        torch.save(state, path)

    def load_memory(self, path: str):
        """Load memory tensors and arrays saved by :meth:`save_memory`."""

        state = torch.load(path, map_location="cpu")

        device = self.memory.device

        def _to_tensor(x, dtype=None):
            if isinstance(x, torch.Tensor):
                return x.to(device=device, dtype=dtype)
            return torch.tensor(x, device=device, dtype=dtype)

        self.memory.data = _to_tensor(state["memory"], dtype=self.memory.dtype)
        self.memory.requires_grad = False

        self.ltm = nn.ParameterList([nn.Parameter(_to_tensor(t, dtype=self.memory.dtype)) for t in state["ltm"]])
        self.ltm_keys = nn.ParameterList([nn.Parameter(_to_tensor(t)) for t in state["ltm_keys"]])
        self.ltm_recall_frequencies = nn.ParameterList([nn.Parameter(_to_tensor(t, dtype=torch.float)) for t in state["ltm_recall_frequencies"]])
        self.ltm_ages = nn.ParameterList([nn.Parameter(_to_tensor(t, dtype=torch.float)) for t in state["ltm_ages"]])

        self.memory_ages = state.get("memory_ages", self.memory_ages)

        self.put_cached_dropped_memory_on_cpu = state.get("put_cached_dropped_memory_on_cpu", True)
        if state.get("cached_dropped_memories") is not None:
            tensor = state["cached_dropped_memories"]
            if self.put_cached_dropped_memory_on_cpu:
                self.cached_dropped_memories = tensor
            else:
                self.cached_dropped_memories = tensor.to(device)
        else:
            self.cached_dropped_memories = None

        self.cached_dropped_memory_ages = state.get("cached_dropped_memory_ages")

        if state.get("cached_dropped_keys") is not None:
            tensor = state["cached_dropped_keys"]
            if self.put_cached_dropped_memory_on_cpu:
                self.cached_dropped_keys = tensor
            else:
                self.cached_dropped_keys = tensor.to(device)
        else:
            self.cached_dropped_keys = None

        self.update_step = state.get("update_step", 0)

    def base_update_memory_with_delta_memory(self, delta_memory, 
                                        cached_contexts_indicators=None, 
                                        is_ltm=False,
                                        retriever_weights=None, 
                                        delta_memory_ages=None, 
                                        return_dropped_memories=False):
        
        if len(delta_memory.shape) == 4:
            delta_memory = delta_memory.detach()[0]

        dropped_memory = [] if return_dropped_memories else None
        dropped_memory_ages = [] if return_dropped_memories else None

        for idx in range(len(self.memory)):

            current_memory = self.memory.data[idx].detach()

            if retriever_weights is not None:

                retriever_labels = retriever_weights[idx] > 0.5
                remaining_indices = torch.where(retriever_labels == 1)[0]

                diff = delta_memory.shape[1] - (len(retriever_labels) - len(remaining_indices))
                if diff > 0:
                    retriever_labels[remaining_indices[:diff]] = False
                    remaining_indices = torch.where(retriever_labels == 1)[0]
                
                indices_to_drop = torch.where(retriever_labels == 0)[0]
                # randomly drop delta_memory.shape[1] indices in indices_to_drop
                remaining_indices = torch.cat([
                    remaining_indices,
                    indices_to_drop[torch.randperm(len(indices_to_drop))[:len(indices_to_drop) - delta_memory.shape[1]]]
                ]).cpu()
                remaining_indices = remaining_indices.sort()[0]
                self.memory.data[idx] = torch.cat([
                    current_memory[remaining_indices],
                    delta_memory[idx]
                ])

                if return_dropped_memories:
                    # TODO: fill this later
                    raise NotImplementedError


            else:
                current_memory, remaining_indices, dropped_indices = self.drop_memory(current_memory, 
                                        delta_memory.shape[1], unsequeezed=False, 
                                        return_remaining_indices=True,
                                        return_dropped_indices=True)
                if return_dropped_memories:
                    dropped_memory.append(
                        self.memory.data[idx][dropped_indices]
                    )
                self.memory.data[idx] = torch.cat([current_memory, delta_memory[idx]], dim=0)

            if is_ltm:

                if len(remaining_indices) == 0:
                    
                    if delta_memory_ages is not None:

                        if return_dropped_memories:
                            dropped_memory_ages.append(self.memory_ages[idx] + 1 + max(delta_memory_ages[idx]))

                        self.memory_ages[idx] = delta_memory_ages[idx]
                    
                    else:

                        raise NotImplementedError

                        if self.update_ltm_mode == 'immediate':
                            self.memory_ages[idx] = np.arange(delta_memory.shape[1])[::-1]
                        else:
                            self.memory_ages[idx] = np.zeros(delta_memory.shape[1])
                    
                    self.memory_recall_frequency[idx] = np.zeros(delta_memory.shape[1])
                    self.memory_position_indicators[idx] = np.ones(delta_memory.shape[1])

                else:
                    
                    # np.array([1,2,3])[torch.tensor([2])] gives 3
                    # np.array([1,2,3])[np.array([2])] gives [3], we need the latter one
                    remaining_indices = np.array(remaining_indices)

                    if return_dropped_memories:
                        dropped_memory_ages.append(self.memory_ages[idx][dropped_indices])

                    if delta_memory_ages is not None:
                        self.memory_ages[idx] = np.concatenate([
                            self.memory_ages[idx][remaining_indices],
                            delta_memory_ages[idx]
                        ])
                        assert delta_memory_ages[idx].shape[0] == delta_memory[idx].shape[0]

                    else:
                        assert delta_memory.shape[1] == self.num_tokens
                        self.memory_ages[idx] = np.concatenate([
                            self.memory_ages[idx][remaining_indices],
                            # np.zeros([delta_memory.shape[1]])
                            np.arange(delta_memory.shape[1])[::-1]
                        ])

            if cached_contexts_indicators is not None:
                if len(cached_contexts_indicators.shape) == 3:
                    # cached_contexts_indicators: [1, L, num_memory_tokens, d]
                    cached_contexts_indicators[:, idx] = torch.cat([
                        cached_contexts_indicators[:, idx][:, remaining_indices],
                        torch.zeros([cached_contexts_indicators.shape[0], delta_memory.shape[1]]).to(cached_contexts_indicators.device)
                    ], dim=1)
                else:
                    # cached_contexts_indicators: [L, num_memory_tokens, d]
                    cached_contexts_indicators[idx] = torch.cat([
                        cached_contexts_indicators[idx][remaining_indices],
                        torch.zeros([delta_memory.shape[1]]).to(cached_contexts_indicators.device)
                    ])

        outputs = ()
        if cached_contexts_indicators is not None:
            outputs = (cached_contexts_indicators,)
        if return_dropped_memories:
            outputs += ((dropped_memory, dropped_memory_ages),)
        
        if len(outputs) == 0:
            return None
        elif len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def update_memory_with_delta_memory(self, 
                                        delta_memory, 
                                        cached_contexts_indicators=None, 
                                        retriever_weights=None, 
                                        delta_memory_ages=None,
                                        dropped_delta_memory=None,
                                        dropped_delta_memory_ages=None):
        
        if delta_memory_ages is not None and dropped_delta_memory_ages is not None:
            if dropped_delta_memory_ages.shape[1] > 0:
                max_delta_memory_age = max(delta_memory_ages.max().item(), dropped_delta_memory_ages.max().item())
            else:
                max_delta_memory_age = delta_memory_ages.max().item()
        else:
            max_delta_memory_age = None
        
        # call the update_memory_with_delta_memory function from the base class
        outputs = self.base_update_memory_with_delta_memory(
                                    delta_memory, 
                                    cached_contexts_indicators, 
                                    is_ltm=True, 
                                    retriever_weights=retriever_weights,
                                    delta_memory_ages=delta_memory_ages,
                                    return_dropped_memories=(self.update_ltm_mode == 'immediate'))

        ages_to_add = self.num_tokens if max_delta_memory_age is None else 1 + max_delta_memory_age

        for idx in range(self.L):
            self.ltm_ages[idx] += ages_to_add

            if self.cached_dropped_memory_ages is not None:
                self.cached_dropped_memory_ages[idx] += ages_to_add

        # update long-term memory each time 
        if cached_contexts_indicators is not None:
            cached_contexts_indicators, (dropped_memories, dropped_memory_ages) = outputs
        else:
            (dropped_memories, dropped_memory_ages) = outputs

        # cat dropped_memories, dropped_delta_memory
        # cat dropped_memory_ages, drop_delta_memory_ages
        if dropped_delta_memory is not None:
            dropped_memories = torch.cat([torch.stack(dropped_memories), dropped_delta_memory[0]], dim=1)
            dropped_memory_ages = np.concatenate([np.stack(dropped_memory_ages) + ages_to_add, dropped_delta_memory_ages], axis=1)
        
        # Accumulate the dropped_memories and dropped_memory_ages and update for onece
        if isinstance(dropped_memories, list):
            dropped_memories = torch.stack(dropped_memories)

        if self.update_step == 0:

            with torch.no_grad():
                cached_dropped_keys = []
                for idx in range(self.L):
                    cached_dropped_keys.append(
                        self.model.layers[idx].self_attn.key_proj(
                            self.model.layers[idx].input_layernorm(
                                dropped_memories[idx]
                            )
                        ).detach().cpu()
                    )
                self.cached_dropped_keys = torch.stack(cached_dropped_keys)

            if self.put_cached_dropped_memory_on_cpu:
                self.cached_dropped_memories = dropped_memories.detach().cpu()
            else:
                self.cached_dropped_memories = dropped_memories.detach()

            self.cached_dropped_memory_ages = dropped_memory_ages

        else:
            
            with torch.no_grad():
                cached_dropped_keys = []
                for idx in range(self.L):
                    cached_dropped_keys.append(
                        self.model.layers[idx].self_attn.key_proj(
                            self.model.layers[idx].input_layernorm(
                                dropped_memories[idx]
                            )
                        ).detach().cpu()
                    )
                cached_dropped_keys = torch.stack(cached_dropped_keys)
                self.cached_dropped_keys = torch.cat([
                    self.cached_dropped_keys,
                    cached_dropped_keys
                ], dim=1)

            # empty torch memory cache
            torch.cuda.empty_cache()

            if self.put_cached_dropped_memory_on_cpu:
                self.cached_dropped_memories = torch.cat([
                    self.cached_dropped_memories, 
                    dropped_memories.detach().cpu()
                ], dim=1)
            else:
                self.cached_dropped_memories = torch.cat([
                    self.cached_dropped_memories, 
                    dropped_memories.detach()
                ], dim=1)
            
            self.cached_dropped_memory_ages = np.concatenate([
                self.cached_dropped_memory_ages,
                dropped_memory_ages
            ], axis=1)

        self.update_step += ages_to_add

        if self.update_step >= self.update_ltm_frequency * self.num_tokens:
            self.merge_cached_memory()
        
        return cached_contexts_indicators
        
    def cat_memory_and_hiddens(self, idx, hidden_states, delta_memory=None, 
                               is_injection=False,
                               cat_to_maximum_memory=False,
                               random_retriever_length=False):
        
        stm = self.get_stm(idx, hidden_states, delta_memory, is_injection, cat_to_maximum_memory)

        ltm_indices = None

        if (not is_injection) and (delta_memory is None or cat_to_maximum_memory):
            ltm, ltm_indices = self.get_ltm(idx, hidden_states, random_retriever_length=random_retriever_length)
            hidden_states = torch.cat([
                ltm.unsqueeze(0),
                stm,
                hidden_states
            ], dim=1)

        else:
            hidden_states = torch.cat([stm, hidden_states], dim=1)

        if self.add_bos_embedding:
            hidden_states = torch.cat([self.bos_embedding[idx].unsqueeze(0).repeat(len(hidden_states), 1, 1), hidden_states], dim=1)
        
        return hidden_states, ltm_indices
    
    def use_decoder_lora(self):
        for _, module in self.named_modules():
            if hasattr(module, "_active_adapter"):
                module._active_adapter = ['decoder_adapter']
    
    def use_encoder_lora(self):
        for _, module in self.named_modules():
            if hasattr(module, "_active_adapter"):
                module._active_adapter = ['default']        

    def inject_memory(self, context_ids, 
                            context_attention_mask=None,
                            delta_memory=None,
                            update_memory=False,
                            use_retriever=False):

        output = self(input_ids=context_ids,
                attention_mask=context_attention_mask,
                delta_memory=delta_memory,
                is_injection=True,
                output_delta_memory=True,
                return_dict=True)

        if update_memory:
            delta_memory = output.delta_memory
            if use_retriever:
                # get retriever_weights
                all_retriever_weights = []
                for idx in range(delta_memory.shape[1]):
                    delta_memory_queries = self.model.layers[idx].self_attn.encoder_query_proj(
                        self.model.layers[idx].input_layernorm(delta_memory[0, idx]))
                    if self.maintain_memory_keys:
                        memory_keys = self.memory_keys[idx]
                    else:
                        memory_keys = self.model.layers[idx].self_attn.key_proj(
                            self.model.layers[idx].input_layernorm(self.memory[idx]))
                    retriever_weights = (delta_memory_queries @ memory_keys.transpose(-2, -1)).sigmoid().mean(dim=0)
                    all_retriever_weights.append(retriever_weights)
                retriever_weights = torch.stack(all_retriever_weights)

            else:
                retriever_weights = None

            self.update_memory_with_delta_memory(delta_memory, retriever_weights=retriever_weights)
            return delta_memory

        else:
            return output.delta_memory


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        delta_memory: Optional[List[List[torch.FloatTensor]]] = None,
        labels: torch.LongTensor = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_delta_memory: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        is_injection: Optional[bool] = None,
        cat_to_maximum_memory: Optional[bool] = False,
        output_retriever_weights: Optional[bool] = False,
        return_full_retriever_weights: Optional[bool] = False,
        random_retriever_length: Optional[bool] = False,
        encoder_query_indices: Optional[List[int]] = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, MemoryLMOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if is_injection is None:
            is_injection = output_delta_memory

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.model.gradient_checkpointing and self.model.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        
        if inputs_embeds is None:

            inputs_embeds = self.model.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # Ensure past_key_values is a Cache object
            past_key_values = DynamicCache()

        # if cache_position is None:
        # TODO: currently ignore cache_position
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        
        if past_seen_tokens > 0:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        else:
            if is_injection:
                cache_position = torch.arange(
                    0, inputs_embeds.shape[1] + self.num_tokens + int(self.add_bos_embedding), device=inputs_embeds.device
                )
            elif delta_memory is not None and delta_memory.shape[2] == self.num_tokens and not cat_to_maximum_memory:
                cache_position = torch.arange(
                    0, inputs_embeds.shape[1] + self.num_tokens + int(self.add_bos_embedding), device=inputs_embeds.device
                )
            else:
                cache_position = torch.arange(
                    0, inputs_embeds.shape[1] + self.num_tokens * (self.num_blocks + self.num_ltm_blocks) + int(self.add_bos_embedding), device=inputs_embeds.device
                )
            
        # TODO: currently ignore position_ids
            
        position_ids = cache_position.unsqueeze(0)

        causal_mask = None

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        all_delta_memory = [] if output_delta_memory else None
        all_retriever_weights = () if output_retriever_weights else None
        all_encoder_retriever_weights = () if (output_retriever_weights and encoder_query_indices is not None) else None
        all_ltm_indices = ()

        if self.add_decoder_lora:

            if is_injection or (delta_memory is not None and delta_memory.shape[2] == self.num_tokens and not cat_to_maximum_memory):
                self.use_encoder_lora()
            else:
                self.use_decoder_lora()

        ltm_indices = None

        for idx, decoder_layer in enumerate(self.model.layers):
            
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if past_key_values is None or past_key_values.get_seq_length(layer_idx=idx) == 0:


                hidden_states, ltm_indices = self.cat_memory_and_hiddens(idx,
                                                hidden_states=hidden_states,
                                                delta_memory=delta_memory,
                                                is_injection=is_injection,
                                                cat_to_maximum_memory=cat_to_maximum_memory,
                                                random_retriever_length=random_retriever_length)

                all_ltm_indices += (ltm_indices,)
                
                prefix_token_length = hidden_states.shape[1] - inputs_embeds.shape[1] if self.initialized else 0

                if is_injection and prefix_token_length > 0:
                    prefix_token_length = min(prefix_token_length, hidden_states.shape[1] - self.num_tokens)

                if is_injection:
                    if self.new_memory_positional_emb.device != hidden_states.device:
                        hidden_states[:, -self.num_tokens:] += self.new_memory_positional_emb.to(hidden_states.device)
                    else:
                        hidden_states[:, -self.num_tokens:] += self.new_memory_positional_emb

            else:
                prefix_token_length = 0

            if self.model.gradient_checkpointing and self.model.training:

                layer_outputs = self.model._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    prefix_token_length,
                    output_retriever_weights,
                )
                
            else:
                
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    prefix_token_length=prefix_token_length,
                    output_retriever_weights=output_retriever_weights,
                    return_full_retriever_weights=return_full_retriever_weights,
                    random_retriever_length=random_retriever_length,
                    encoder_query_indices=encoder_query_indices[idx] if encoder_query_indices is not None else None,
                    ltm_length=self.num_ltm_blocks * self.num_tokens,
                    training=training,
                )

            hidden_states = layer_outputs[0]
            if output_delta_memory:
                all_delta_memory.append(hidden_states[:, -self.num_tokens:])

            hidden_states = hidden_states[:, -input_ids.shape[1]:]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_retriever_weights:
                if encoder_query_indices is not None:
                    retriever_weights = layer_outputs[-2]
                    encoder_retriever_weights = layer_outputs[-1]
                    all_encoder_retriever_weights += (encoder_retriever_weights,)
                else:
                    retriever_weights = layer_outputs[-1]
                if retriever_weights is not None:
                    all_retriever_weights += (retriever_weights,)
            
        hidden_states = self.model.norm(hidden_states)
            
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if output_delta_memory:

            if all_delta_memory[0].device != all_delta_memory[-1].device:
                assert not self.training
                device = all_delta_memory[0].device
                all_delta_memory = [x.to(device) for x in all_delta_memory]
                delta_memory = torch.stack(all_delta_memory, dim=0).transpose(0, 1)

            else:
                delta_memory = torch.stack(all_delta_memory, dim=0).transpose(0, 1)

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            return tuple(v for v in [loss, logits, hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return MemoryLMOutputWithPastAndCrossAttentions(
            loss=loss,
            logits=logits,
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            delta_memory=delta_memory,
            retriever_weights=all_retriever_weights if (all_retriever_weights is not None and len(all_retriever_weights) > 0) else None,
            encoder_retriever_weights=all_encoder_retriever_weights if (all_encoder_retriever_weights is not None and len(all_encoder_retriever_weights) > 0) else None,
            ltm_indices=all_ltm_indices if (len(all_ltm_indices) > 0 and all_ltm_indices[0] is not None) else None
        )

    def drop_memory(self, current_memory, drop_length=None, unsequeezed=True, return_remaining_indices=False, return_dropped_indices=False):

        if unsequeezed:

            perm_indices = torch.randperm(current_memory.shape[1])

            if drop_length is None:
                remaining_indices = perm_indices[:current_memory.shape[1] - int(current_memory.shape[1] * (1 / self.num_blocks))]
            else:
                remaining_indices = perm_indices[:current_memory.shape[1] - drop_length]
            
            # sort remaining_indices to make sure it is in ascending order
            remaining_indices = remaining_indices.sort()[0]
            dropped_indices = perm_indices[len(remaining_indices):]
            dropped_indices = dropped_indices.sort()[0]

            current_memory = current_memory[:, remaining_indices, :]
            
        else:

            perm_indices = torch.randperm(current_memory.shape[0])

            if drop_length is None:
                remaining_indices = perm_indices[:current_memory.shape[0] - int(current_memory.shape[0] * (1 / self.num_blocks))]
            else:
                remaining_indices = perm_indices[:current_memory.shape[0] - drop_length]
            
            # sort remaining_indices to make sure it is in ascending order
            remaining_indices = remaining_indices.sort()[0]
            dropped_indices = perm_indices[len(remaining_indices):]
            dropped_indices = dropped_indices.sort()[0]

            current_memory = current_memory[remaining_indices, :]
    
        if return_remaining_indices and return_dropped_indices:
            return current_memory, remaining_indices, dropped_indices
        if return_remaining_indices:
            return current_memory, remaining_indices
        elif return_dropped_indices:
            return current_memory, dropped_indices
        else:
            return current_memory

    # The followings are the functions for long-term memory
    def get_stm(self, 
                idx,
                hidden_states,
                delta_memory=None,
                is_injection=False,
                cat_to_maximum_memory=False):
        
        if delta_memory is None or len(delta_memory) == 0:
            if is_injection:
                cur_memory = self.memory[idx][ - self.num_tokens:].unsqueeze(0).repeat(len(hidden_states), 1, 1)
            else:
                cur_memory = self.memory[idx].unsqueeze(0).repeat(len(hidden_states), 1, 1)
        else:
            cur_memory = delta_memory[:, idx]
            if (not is_injection) and cat_to_maximum_memory:

                old_memory = self.memory[idx].detach().unsqueeze(0).repeat(len(hidden_states), 1, 1) # detach might be unnecessary, but just to make sure

                # put on cuda
                if old_memory.device != hidden_states.device:
                    old_memory = old_memory.to(hidden_states.device)

                sampled_indices = torch.randperm(old_memory.shape[1])[:old_memory.shape[1] - cur_memory.shape[1]]
                # sort sampled_indices to make sure it is in ascending order
                sampled_indices = sampled_indices.sort()[0]
                old_memory = old_memory[:, sampled_indices, :]
                cur_memory = torch.cat([
                    old_memory,
                    cur_memory,
                ], dim=1)
            
        return cur_memory

    def get_ltm(self, idx, hidden_states, random_retriever_length=False):

        num_ltm_tokens = self.num_ltm_tokens if hasattr(self, "num_ltm_tokens") else self.num_ltm_blocks * self.num_tokens

        # get ltm_keys if ltm_keys are None
        if self.ltm_keys[idx] is None:

            with torch.no_grad():

                ltm = self.ltm[idx]
                tmp_ltm_keys = []
                batch_size = 64

                for batch_ltm in torch.split(ltm, batch_size):
                    batch_ltm = batch_ltm.to(hidden_states.device).to(hidden_states.dtype)
                    ltm_keys = self.model.layers[idx].self_attn.key_proj(self.model.layers[idx].input_layernorm(batch_ltm))
                    tmp_ltm_keys.append(ltm_keys)
                tmp_ltm_keys = torch.cat(tmp_ltm_keys, dim=0)

            tmp_ltm_keys = tmp_ltm_keys.detach().cpu()
            self.ltm_keys[idx] = tmp_ltm_keys

        with torch.no_grad():

            if len(self.ltm_keys[idx]) < num_ltm_tokens:

                indices = torch.tensor([])
                while len(indices) < num_ltm_tokens:
                    indices = torch.cat([
                        torch.arange(len(self.ltm_keys[idx])),
                        indices
                    ])
                indices = indices[-num_ltm_tokens:].to(torch.int)
            
            else:
                
                if random_retriever_length:
                    length = torch.randint(1, hidden_states.size(1), (1,)).item()
                    hidden_states = hidden_states[:, :length, :]

                queries = self.model.layers[idx].self_attn.query_proj(self.model.layers[idx].input_layernorm(hidden_states[0]))
                predictions = (queries @ self.ltm_keys[idx].to(queries.device).transpose(-2, -1)).sigmoid().mean(dim=0)

                if self.cached_dropped_memories is not None:
                    cached_predictions = (queries @ self.cached_dropped_keys[idx].to(queries.device).transpose(-2, -1)).sigmoid().mean(dim=0)
                    predictions = torch.cat([predictions, cached_predictions], dim=0)

                indices = torch.topk(predictions, k=num_ltm_tokens).indices
                indices = torch.sort(indices)[0].cpu()

        if self.cached_dropped_memories is None:
            ages = self.ltm_ages[idx][indices.detach().cpu()]
            indices = indices[np.argsort(ages)[::-1].copy()]
            x = self.ltm[idx][indices.detach().cpu()].to(hidden_states.device)
            return x.detach(), indices.detach().cpu()
        
        else:

            with torch.no_grad():

                ltm_indices = indices[torch.where(indices < self.ltm[idx].shape[0])[0]]
                dropped_indices = indices[len(ltm_indices):] - self.ltm[idx].shape[0]

                ltm_ages = self.ltm_ages[idx][ltm_indices.detach().cpu().numpy()]
                dropped_ages = self.cached_dropped_memory_ages[idx][np.array(dropped_indices.detach().cpu())]

                ltm_x = self.ltm[idx][ltm_indices].to(hidden_states.device)
                dropped_x = self.cached_dropped_memories[idx][dropped_indices].to(hidden_states.device)

                if len(dropped_ages) > 0 and len(ltm_ages) > 0:
                    all_ages = np.concatenate([ltm_ages, dropped_ages])
                else:
                    all_ages = ltm_ages if len(ltm_ages) > 0 else dropped_ages

                all_x = torch.cat([ltm_x, dropped_x], dim=0)

                all_x = all_x[np.argsort(all_ages)[::-1].copy()]

            return all_x.detach(), ltm_indices
    
    def update_recall_frequency(self, idx, hidden_states):

        with torch.no_grad():
            queries = self.model.layers[idx].self_attn.encoder_query_proj(self.model.layers[idx].input_layernorm(hidden_states[0]))
            keys = self.model.layers[idx].self_attn.key_proj(self.model.layers[idx].input_layernorm(self.memory[idx]))
            indices = torch.where((queries @ keys.transpose(-2, -1)).sigmoid().mean(dim=0) > 0.5)[0]
            self.memory_recall_frequency[idx][indices.detach().cpu().numpy()] += 1
    
    def update_ltm(self, dropped_memories=None, dropped_memory_ages=None, device=None, cached_dropped_keys=None):

        with torch.no_grad():

            # update ltm according to memory_recall_frequency
            for idx in range(len(self.memory)):

                current_memory = dropped_memories[idx]
                self.ltm[idx] = torch.cat([
                    self.ltm[idx],
                    current_memory.detach().cpu()
                ])
                assert self.initial_rf_when_moving_stm_to_ltm is not None
                self.ltm_recall_frequencies[idx] = np.concatenate(
                    [self.ltm_recall_frequencies[idx],
                    np.ones(current_memory.shape[0]) * self.initial_rf_when_moving_stm_to_ltm]
                )

                if cached_dropped_keys is not None:

                    self.ltm_keys[idx] = torch.cat([
                        self.ltm_keys[idx],
                        cached_dropped_keys[idx]
                    ])
                
                else:

                    self.ltm_keys[idx] = torch.cat([
                        self.ltm_keys[idx],
                        self.model.layers[idx].self_attn.key_proj(
                            self.model.layers[idx].input_layernorm(
                                current_memory.to(device) if device is not None else current_memory
                            )
                        ).detach().cpu()
                    ])

                self.ltm_ages[idx] = np.concatenate([
                    self.ltm_ages[idx].astype(int),
                    dropped_memory_ages[idx]
                ])

                self.ltm_recall_frequencies[idx] -= self.decay_frequency
                indices = np.where(self.ltm_recall_frequencies[idx] > 0.01)[0] # sometimes it may be "2.0539126e-15", using 0.01 to filter out these cases
                if len(indices) > self.num_ltm_blocks * self.num_tokens:
                    self.ltm[idx] = self.ltm[idx][indices]
                    self.ltm_keys[idx] = self.ltm_keys[idx][indices]
                    self.ltm_recall_frequencies[idx] = self.ltm_recall_frequencies[idx][indices]
                    self.ltm_ages[idx] = self.ltm_ages[idx][indices]

                else:
                    self.ltm_recall_frequencies[idx] += self.decay_frequency

