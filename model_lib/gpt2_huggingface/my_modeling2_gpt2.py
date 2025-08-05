""" NOTE: 
    1) Dropout removed tentatively
    2) Add Simplified GPT2 model 
"""
# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch OpenAI GPT-2 model."""

import logging
import math
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from .activations import ACT2FN
from .configuration_gpt2 import GPT2Config
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_utils import Conv1D, PreTrainedModel, SequenceSummary, prune_conv1d_layer

# =======================
# === for GL version ====
# =======================
import time
from collections import OrderedDict, deque

logger = logging.getLogger(__name__)

GPT2_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "gpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin",
    "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-pytorch_model.bin",
    "gpt2-large": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-pytorch_model.bin",
    "gpt2-xl": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-pytorch_model.bin",
    "distilgpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-pytorch_model.bin",
}


def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()
        self.output_attentions = config.output_attentions

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        # self.attn_dropout = nn.Dropout(config.attn_pdrop)
        # self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd : ns, :ns]
        w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        # w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        # a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        # self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2 
        # return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
        output_attn = self.attn(
            self.ln_1(x), layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask
        )
        a = output_attn[0]  # output_attn: a, present, (attentions)

        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m

        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)


class GPT2PreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = GPT2Config
    pretrained_model_archive_map = GPT2_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


GPT2_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.GPT2Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

GPT2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length if `past` is None else 1
            Indices of input sequence tokens in the vocabulary.
            If using `past` as an input make sure that `input_ids` are those of the last position.

            Indices can be obtained using :class:`transformers.GPT2Tokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__

        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `past` output below). Can be used to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`, `optional`, defaults to :obj:`None`):
            `input_ids_length` = `sequence_length if `past` is None else 1
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            If using `past` as an input make sure that `token_type_ids` correspond to the `input_ids` of the last position.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        input_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)
class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.output_past = config.output_past

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        # self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import GPT2Tokenizer, GPT2Model
        import torch

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        # hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[i]
            )

            hidden_states, present = outputs[:2]
            if self.output_past:
                presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_past:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)


@add_start_docstrings(
    """The GPT2 Model transformer with a language modeling head on top
    (linear layer with weights tied to the input embeddings). """,
    GPT2_START_DOCSTRING,
)
class GPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {"input_ids": input_ids, "past": past}

    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when ``labels`` is provided)
            Language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2LMHeadModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

        """
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)


@add_start_docstrings(
    """The GPT2 Model transformer with a language modeling and a multiple-choice classification
    head on top e.g. for RocStories/SWAG tasks. The two heads are two linear layers.
    The language modeling head has its weights tied to the input embeddings,
    the classification head takes as input the input of a specified classification token index in the input sequence).
""",
    GPT2_START_DOCSTRING,
)
class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
    ):
        r"""
        mc_token_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input)
            Index of the classification token in each input sequence.
            Selected in the range ``[0, input_ids.size(-1) - 1[``.
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`)
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        mc_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`, `optional`, defaults to :obj:`None`)
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``lm_labels`` is provided):
            Language modeling loss.
        mc_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`multiple_choice_labels` is provided):
            Multiple choice classification loss.
        lm_prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

        # Add a [CLS] to the vocabulary (we should train it also!)
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
        print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary

        choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        encoded_choices = [tokenizer.encode(s) for s in choices]
        cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

        input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
        mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

        outputs = model(input_ids, mc_token_ids=mc_token_ids)
        lm_prediction_scores, mc_prediction_scores = outputs[:2]

        """
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        outputs = (lm_logits, mc_logits) + transformer_outputs[1:]
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
            outputs = (loss,) + outputs
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (lm loss), (mc loss), lm logits, mc logits, presents, (all hidden_states), (attentions)


class GPT2Embeddings(torch.nn.Module):
    def __init__(self, config):
        super(GPT2Embeddings, self).__init__()
        self.wte = torch.nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = torch.nn.Embedding(config.n_positions, config.n_embd)
        # self.drop = torch.nn.Dropout(config.embd_pdrop) 
        self.config = config
        
    def forward(self, input_ids): 
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
         
        past_length = 0
        
        device = input_ids.device
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        # hidden_states = self.drop(hidden_states) 

        return hidden_states

class GPT2Layer(torch.nn.Module):
    def __init__(self, config):
        super(GPT2Layer, self).__init__()
        self.h = Block(config.n_ctx, config, scale=True)
        self.config = config
        
    def forward(self, hidden_states):
        outputs = self.h(hidden_states,
                        layer_past=None,
                        attention_mask=None,
                        head_mask=None)
        hidden_states, _ = outputs[:2]

        return hidden_states
    

# -------------------- gl version -----------------------------------

# 从cpu_cache中返回张量。若没找到对应的张量,创建一个固定内存中的全零张量并返回
def _get_item_from_cpu_cache(_cpu_cache, key, value=None):
    # 若cpu_cache中保存了给定key对应的张量,直接返回
    if key in _cpu_cache and _cpu_cache[key] is not None:
        ## ----- nvme -----
        #if isinstance(_cpu_cache[key], MockTensor):
        #    #import offloading_utils
        #    #_mock_tensor = _cpu_cache[key]
        #    #_cpu_cache[key] = offloading_utils.load(_cpu_cache[key].loc)[0]
        #    #_cpu_cache[key].mock_tensor = _mock_tensor

        #    _cpu_cache[key] = ray.get(_cpu_cache[key].loc)
        ## ----------------
        return _cpu_cache[key]

    # 若没找到对应的张量,创建一个固定内存中的全零张量并返回
    _cpu_cache[key] = torch.zeros(
        value.size(),
        dtype=value.dtype,
        layout=value.layout,
        device=torch.device("cpu"),
        pin_memory=True,  # (torch.cuda.is_available() and not value.is_sparse)
    ).to("cpu", non_blocking=True)
    return _cpu_cache[key]

def _move_item_to_nvme(_cpu_cache, key, value=None):
    if key in _cpu_cache and _cpu_cache[key] is not None:
        ## ----- nvme -----
        #if not isinstance(_cpu_cache[key], MockTensor): 
        #    #import offloading_utils
        #    if hasattr(_cpu_cache[key], 'mock_tensor'):
        #        #mock_tensor = _cpu_cache[key].mock_tensor
        #        #del _cpu_cache[key].mock_tensor
        #        #offloading_utils.save([_cpu_cache[key]], mock_tensor.loc)
        #        #_cpu_cache[key] = mock_tensor
        #        
        #        r = ray.put(_cpu_cache[key])
        #        mock_tensor = MockTensor(_cpu_cache[key], r)
        #        _cpu_cache[key] = mock_tensor
        #    else:
        #        #saved_name = '/tmp/' + str(id(_cpu_cache[key])) + str(random.random()) + '.pt'
        #        #mock_tensor = MockTensor(_cpu_cache[key], saved_name)
        #        #offloading_utils.save([_cpu_cache[key]], saved_name)
        #        #_cpu_cache[key] = mock_tensor
        #        
        #        r = ray.put(_cpu_cache[key])
        #        mock_tensor = MockTensor(_cpu_cache[key], r)
        #        _cpu_cache[key] = mock_tensor
        ## ----------------
        pass 

    return

class GL_GPT2Layer(GPT2Layer):
    def __init__(self, config, _gl_cuda_cache_queue=None):
        assert (
            _gl_cuda_cache_queue is not None
        ), "Please initialize GL_GPT2Layer using _cuda_cache_queue"
        self._gl_cuda_cache_queue = _gl_cuda_cache_queue

        super(GL_GPT2Layer, self).__init__(config)

        # 存储在 CPU 端的参数、梯度和缓冲区
        self._gl_cpu_version_items = OrderedDict()

        # 将cpu上的 参数\梯度\缓冲区 ,拷贝到cpu的pinned-memory上,其原本的指向也被重定向到pinned memory上
        for module_name, module in self.named_modules():
            print(f"正在执行module:{module_name}")
            for param_name, param in module._parameters.items():
                print(f"正在执行参数 (param_name:{param_name}) 的迁移工作")
                if param is not None:
                    _pkey = module_name + param_name

                    self._gl_cpu_version_items[_pkey] = torch.empty(
                        param.size(),
                        dtype=param.dtype,
                        layout=param.layout,
                        device=torch.device("cpu"),
                        pin_memory=True,  # (torch.cuda.is_available() and not tensor.is_sparse),
                    ).copy_(param, non_blocking=False)
                    # 将原始参数的数据指向这个新的 CPU 张量
                    param.data = self._gl_cpu_version_items[_pkey]

                    _gkey = _pkey + ".grad"
                    if param.grad is not None:
                        self._gl_cpu_version_items[_gkey] = (
                            torch.empty(
                                param.grad.size(),
                                dtype=param.grad.dtype,
                                layout=param.grad.layout,
                                device=torch.device("cpu"),
                                pin_memory=True,  # (torch.cuda.is_available() and not tensor.is_sparse)
                            )
                            .copy_(
                                param.grad,
                                non_blocking=False,
                            )
                            .requires_grad_(param.grad.requires_grad)
                        )
                        # 将原始梯度的数据指向这个新的 CPU 张量
                        param.grad.data = self._gl_cpu_version_items[_gkey]
                    
                    # 如果参数没有梯度，则创建一个全零的张量
                    else:
                        self._gl_cpu_version_items[_gkey] = torch.empty(
                            param.size(),
                            dtype=param.dtype,
                            layout=param.layout,
                            device=torch.device("cpu"),
                            pin_memory=True,  # (torch.cuda.is_available() and not tensor.is_sparse)
                        ).copy_(torch.zeros_like(param), non_blocking=False)

                        param.grad = self._gl_cpu_version_items[_gkey]

            for buffer_name, buffer in module._buffers.items():
                print(f"正在执行缓冲区 (buffer_name:{buffer_name}) 的迁移工作")
                if buffer is not None:
                    _bkey = "buffer_" + module_name + buffer_name

                    self._gl_cpu_version_items[_bkey] = torch.empty(
                        buffer.size(),
                        dtype=buffer.dtype,
                        layout=buffer.layout,
                        device=torch.device("cpu"),
                        pin_memory=True,  # (torch.cuda.is_available() and not tensor.is_sparse)
                    ).copy_(buffer, non_blocking=False)
        return
    
    def compute_should_use_set_data(self, tensor, tensor_applied):
        # if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
        #     # If the new tensor has compatible tensor type as the existing tensor,
        #     # the current behavior is to change the tensor in-place using `.data =`,
        #     # and the future behavior is to overwrite the existing tensor. However,
        #     # changing the current behavior is a BC-breaking change, and we want it
        #     # to happen in future releases. So for now we introduce the
        #     # `torch.__future__.get_overwrite_module_params_on_conversion()`
        #     # global flag to let the user control whether they want the future
        #     # behavior of overwriting the existing tensor or not.
        #     return not torch.__future__.get_overwrite_module_params_on_conversion()
        # else:
        #     return False

        """
        If return False, would be error on "grad of a different type"
        todo. @gl
        """
        return True

    # 1.将transformer layer中所有层的 参数\梯度\buffer 复制到cpu cache上(pinned memory上的tensor)
    # 2.同时将_cuda_cache_unit（对当前transformer layer cuda上参数\梯度\buffer的指向）放回到cuda cache队列中
    @torch.no_grad()
    def cpu(self, non_blocking=True):
        # already cached in cpu-side
        # 📌如果没有 _gl_cuda_cache_unit 这个属性，说明还没执行过gpu()方法，参数什么的本来就在cpu上，直接返回
        if not hasattr(self, "_gl_cuda_cache_unit"):
            return

        _cuda_cache_unit = self._gl_cuda_cache_unit

        # 将给定module的参数\梯度\buffer复制到其对应的cpu cache中(pinned memory中的tensor)
        # 同时保留一个对cuda上参数\梯度\buffer的指向,放在_cuda_cache_unit中
        def _apply_cpu(module_name, module, cpu_version_items):
            for param_name, param in module._parameters.items():
                if param is None:
                    continue
                if param.grad is None and str(param.device) == "cpu":
                    continue
                if (
                    param.grad is not None
                    and str(param.grad.device) == "cpu"
                    and str(param.device) == "cpu"
                ):
                    continue

                _pkey = module_name + param_name

                # 使用_pkey拿取cpu上的tensor,并将参数(param)复制到cpu的pinned memory上
                # 设置复制后的张量的 requires_grad 属性与 param 相同(保留其requires_grad属性)
                param_applied = (
                    _get_item_from_cpu_cache(self._gl_cpu_version_items, _pkey, param)
                    .copy_(param, non_blocking=non_blocking)
                    .requires_grad_(param.requires_grad)
                )
                # 该函数就是pass
                _move_item_to_nvme(self._gl_cpu_version_items, _pkey)

                # 若参数的梯度为空,将原本的数据(.data)存到cuda cache中
                # 将装着param的pinned memory赋给param.data,即参数现在的数据指向cpu了
                if param.grad is None:
                    param_applied.grad = None

                    # 该函数直接返回true
                    use_shallow_copy = self.compute_should_use_set_data(
                        param, param_applied
                    )

                    #
                    if use_shallow_copy:
                        # 保留一个对cuda上buffer的指向,放在_cuda_cache_unit中
                        _cuda_cache_unit[_pkey] = param.data
                        # 将装着param的pinned memory赋给param.data,即参数现在的数据指向cpu了
                        param.data = param_applied
                        param.grad = None
                    else:
                        assert param.is_leaf
                        _cuda_cache_unit[_pkey] = module._parameters[param_name]
                        module._parameters[param_name] = param_applied.requires_grad_(
                            param.requires_grad
                        )

                # 否则,多出一步复制梯度的次操作,即将参数的梯度复制到pinned memory的tensor上
                # 并将pinned memory上的参数的梯度指向pinned memory上的梯度
                else:
                    _gkey = _pkey + ".grad"

                    # 使用_gkey拿取cpu上的tensor,并将梯度复制到cpu的pinned memory上
                    # 设置复制后的张量的 requires_grad 属性与 param.grad 相同(保留其requires_grad属性)
                    grad_applied = (
                        _get_item_from_cpu_cache(
                            self._gl_cpu_version_items, _gkey, param.grad
                        )
                        .copy_(param.grad)  # todo? @gl copy_ or add_
                        .requires_grad_(param.grad.requires_grad)
                    )

                    # print("grad_applied", grad_applied.device)
                    # grad_applied = grad_applied.cpu()
                    # print("grad_applied", grad_applied.device)

                    use_shallow_copy = self.compute_should_use_set_data(
                        param.grad, grad_applied
                    )

                    # param_applied.grad = grad_applied

                    if use_shallow_copy:
                        _cuda_cache_unit[_pkey] = param.data
                        _cuda_cache_unit[_gkey] = param.grad.data

                        # 该pinned memory上的tensor的grad现在指向pinned memory上的梯度(tensor)
                        param_applied.grad = grad_applied
                        param.data = param_applied
                        # param.grad.data = grad_applied

                    else:
                        assert param.is_leaf
                        _cuda_cache_unit[_pkey] = module._parameters[param_name]
                        _cuda_cache_unit[_gkey] = module._parameters[param_name].grad

                        module._parameters[param_name] = param_applied.requires_grad_(
                            param.requires_grad
                        )
                        module._parameters[
                            param_name
                        ].grad = grad_applied.requires_grad_(param.grad.requires_grad)

                    _move_item_to_nvme(self._gl_cpu_version_items, _gkey)

            for buffer_name, buffer in module._buffers.items():
                if buffer is None:
                    continue

                # 若buffer的设备是cpu,说明已经在cpu上了,直接略过
                if str(buffer.device) == "cpu":
                    continue

                _bkey = "buffer_" + module_name + buffer_name

                # 保留一个对cuda上buffer的指向
                _cuda_cache_unit[_bkey] = module._buffers[buffer_name]

                # 将buffer复制到cpu cache上(pinned memory tensor)
                module._buffers[buffer_name] = _get_item_from_cpu_cache(
                    self._gl_cpu_version_items, _bkey, buffer
                ).copy_(buffer, non_blocking=non_blocking)

                _move_item_to_nvme(self._gl_cpu_version_items, _bkey)

        # 将transformer layer中所有层的 参数\梯度\buffer 复制到cpu cache上(pinned memory上的tensor)
        for module_name, module in self.named_modules():
            _apply_cpu(module_name, module, self._gl_cpu_version_items)

        # for key, value in self._gl_cpu_version_items.items():
        #     print(f" -- debug: _gl_cpu_version_items after apply_cpu: {key}, {value.device}")

        # return cuda_cache as a unit
        # 将暂存的对cuda上参数\梯度\buffer的指向（一个字典），添加到self._gl_cuda_cache_queue，
        # 📌即将cuda cache添加回cuda cache队列中，留给其他层复制到GPU时用
        self._gl_cuda_cache_queue.append(_cuda_cache_unit)

        # remove the cuda cache from itself
        # 删除该layer的 _gl_cuda_cache_unit 属性，表明该层现在在cpu上
        del self._gl_cuda_cache_unit

        pass

    # 将transformer layer中所有层的 参数\梯度\buffer 从cpu复制到GPU
    # 分析：
    @torch.no_grad()
    def cuda(self, device=None, non_blocking=True):
        # already cached in cuda-side
        # 若已存在属性 _gl_cuda_cache_unit 直接返回，说明已经缓存在GPU上了
        if hasattr(self, "_gl_cuda_cache_unit"):
            return

        # current cuda device
        device = torch.cuda.current_device() if device is None else device

        # pick up an allocated cuda_cuda_unit from cuda_cache_queue
        # 从 _gl_cuda_cache_queue 弹出一个字典，装着对当前layer在GPU上 参数、梯度、buffer 的指向
        _cuda_cache_unit = self._gl_cuda_cache_queue.pop()

        # 创建_gl_cuda_cache_unit属性，表明该层现在在GPU上
        self._gl_cuda_cache_unit = _cuda_cache_unit

        # for key, value in self._gl_cpu_version_items.items():
        #     print(f" -- debug: _gl_cpu_version_items before: {key}, {value.device}")

        def _apply_cuda(module_name, module, cpu_version_items):
            for param_name, param in module._parameters.items():
                if param is None:
                    continue

                _pkey = module_name + param_name

                # 1.复制参数到GPU
                # 若GPU上没有对应该参数的cache（一个GPU上的tensor），从cpu cache中返回一个张量并移动到GPU上
                if _pkey not in _cuda_cache_unit or _cuda_cache_unit[_pkey] is None:
                    print(f" --> _apply_cuda: failed to reuse cuda tensor: {_pkey}")

                    # 从cpu_cache中返回张量（若没找到对应的张量,创建一个固定内存中的全零张量并返回），并移动到GPU上
                    param_applied = _get_item_from_cpu_cache(
                        cpu_version_items, _pkey).to(
                            device,
                            non_blocking=non_blocking,
                            copy=True,
                        )
                    
                # 否则，说明有，无需进行移动操作了，直接将cpu cache中的参数复制到GPU的张量上
                # 分析：相对于if，相当于省了 to 这一步
                else:
                    param_applied = _cuda_cache_unit[_pkey].copy_(
                        _get_item_from_cpu_cache(
                            cpu_version_items, _pkey), non_blocking=non_blocking
                    )
                    assert (
                        str(param_applied.device) != "cpu"
                    ), "the tensor should be on cuda device after invoking _apply_cuda()"

                # 保留参数原本的 requires_grad 属性
                param_applied = param_applied.requires_grad_(param.requires_grad)

                # 2.复制梯度到GPU
                # 
                if param.grad is None:
                    param_applied.grad = None

                    # 该函数直接返回true
                    use_shallow_copy = self.compute_should_use_set_data(
                        param, param_applied
                    )
                    if use_shallow_copy:
                        param.data = param_applied
                        param.grad = None
                    else:
                        assert param.is_leaf
                        module._parameters[param_name] = param_applied.requires_grad_(
                            True
                        )

                else:
                    _gkey = _pkey + ".grad"

                    if _gkey not in _cuda_cache_unit or _cuda_cache_unit[_gkey] is None:
                        print(f" --> _apply_cuda: failed to reuse cuda tensor: {_gkey}")

                        grad_applied = _get_item_from_cpu_cache(
                            cpu_version_items, _gkey).to(
                                device,
                                non_blocking=non_blocking,
                                copy=True,
                            )
                    else:
                        # print(f"before cpu_version_items {_gkey},", cpu_version_items[_gkey].device,
                        #     id(cpu_version_items[_gkey]) == id(self._gl_cpu_version_items[_gkey]))

                        # 将cpu上的梯度复制到GPU的tensor上
                        grad_applied = _cuda_cache_unit[_gkey].copy_(
                            _get_item_from_cpu_cache(
                                cpu_version_items, _gkey), non_blocking=non_blocking
                        )

                        # print(f"after {_gkey},", cpu_version_items[_gkey].device,
                        #     id(cpu_version_items[_gkey]) == id(self._gl_cpu_version_items[_gkey]))

                        assert (
                            str(grad_applied.device) != "cpu"
                        ), "the gradient should be on cuda device after invoking _apply_cuda()"

                        # param_applied.grad = grad_applied

                        use_shallow_copy = self.compute_should_use_set_data(
                            param, param_applied
                        )
                        if use_shallow_copy:
                            param_applied.grad = grad_applied
                            param.data = param_applied

                            # the big buggy error: it would make cpu_verion grads be located at CUDA side
                            # why. how to address it?
                            # todo. @gl.
                            # param.grad.data = grad_applied
                        else:
                            assert param.is_leaf
                            module._parameters[
                                param_name
                            ] = param_applied.requires_grad_(param.requires_grad)
                            module._parameters[
                                param_name
                            ].grad = grad_applied.requires_grad_(
                                param.grad.requires_grad
                            )

            # 3.复制buffer到GPU
            for buffer_name, buffer in module._buffers.items():
                if buffer is None:
                    continue

                _bkey = "buffer_" + module_name + buffer_name

                if _bkey not in _cuda_cache_unit or _cuda_cache_unit[_bkey] is None:
                    print(
                        f" --> _apply_cuda: failed to reuse cuda tensor for buffers: {_bkey}"
                    )

                    module._buffers[buffer_name] = _get_item_from_cpu_cache(
                        cpu_version_items, _bkey).to(
                            device,
                            non_blocking=non_blocking,
                            copy=True,
                        )
                else:
                    module._buffers[buffer_name] = _cuda_cache_unit[_bkey].copy_(
                        _get_item_from_cpu_cache(
                            cpu_version_items, _bkey), non_blocking=non_blocking
                    )
                    assert (
                        str(module._buffers[buffer_name].device) != "cpu"
                    ), "the buffers should be on cuda device after invoking _apply_cuda()"

        # 对transformer layer中的每一层，将参数、梯度、buffer从cpu复制到gpu上
        for module_name, module in self.named_modules():
            _apply_cuda(module_name, module, self._gl_cpu_version_items)

        # for key, value in self._gl_cpu_version_items.items():
        #     print(f" -- debug: _gl_cpu_version_items after apply_cuda: {key}, {value.device}")

        pass

# my gl transformer
class Transformer(torch.nn.Module):
    def __init__(self, config, gl_window_size):
        super(Transformer, self).__init__()

        # Transformer layers.
        # avoid too much cuda allocation
        #
        # 初始化一个 CUDA 缓存队列，用于减少 CUDA 内存分配
        # 队列最大长度（maxlen）：窗口数量×2、模型层数之间的最小值
        _cuda_cache_queue = deque(maxlen=min(gl_window_size * 2, config.n_layer))

        def _build_layer(config):
            layer = GL_GPT2Layer(config, _gl_cuda_cache_queue=_cuda_cache_queue)
            return layer

        # 创建 config.n_layer 个 transformer 层
        self.layers = torch.nn.ModuleList([ _build_layer(config) 
                                            for _ in range(config.n_layer) ])

        # 创建cuda cache
        if True:
            device = torch.cuda.current_device()
            # window size×2 和 层数之间的最小值
            _max_num_units = min(gl_window_size * 2, config.n_layer)

            for i in range(_max_num_units):
                # 遍历 _max_num_units，为每个unit 创建一个有序字典
                unit = OrderedDict()
                # ❓代码写错了？
                # 分析📌：应该没写错，这里只是在GPU上创建相同大小的tensor，由于transformer每个层的结构相同，命名也相同，
                #         这里就直接用第0层创建所有的cache了（2倍窗口大小个层的cache），反正所有层都能用相同的名字取出cache
                #         再一个，把0改成i，第一个transformer层先进行cuda()操作，弹出的却是最后一个层（2倍窗口大小的最后一层）
                #         的cuda cache，明显对应不上
                #         📌所以核心在于，这里只是创建cache，不是直接拷贝这么多个层的参数等到GPU，否则大小只应该是window size
                #         而不是两倍
                _layer = self.layers[0]  # every layer is of the same structure
                # _layer = self.layers[i]

                for module_name, module in _layer.named_modules():
                    for param_name, param in module._parameters.items():
                        _pkey = module_name + param_name

                        # 若参数不为空，将其从cpu移动到gpu
                        if param is not None:
                            unit[_pkey] = (
                                _layer._gl_cpu_version_items[_pkey]
                                .to(device)
                                .requires_grad_(param.requires_grad)
                                .cuda()
                            )

                        _gkey = _pkey + ".grad"

                        # 若梯度不为空，将其从cpu移动到gpu
                        if param.grad is not None:
                            unit[_gkey] = (
                                _layer._gl_cpu_version_items[_gkey]
                                .to(device)
                                .requires_grad_(param.grad.requires_grad)
                                .cuda()
                            )
                        else:
                            unit[_gkey] = _layer._gl_cpu_version_items[_gkey].to(device)

                    # 如果缓冲区存在，则将其从 CPU 移动到 GPU
                    for buffer_name, buffer in module._buffers.items():
                        if buffer is not None:
                            _bkey = "buffer_" + module_name + buffer_name
                            unit[_bkey] = _layer._gl_cpu_version_items[_bkey].to(device)

                # 📌将指向GPU上参数、梯度、buffer的有序字典添加到当前 layer 的 _gl_cuda_cache_queue 中(一个有序字典)
                _layer._gl_cuda_cache_queue.append(unit)



class GPT2LayerNorm(torch.nn.Module):
    def __init__(self, config):
        super(GPT2LayerNorm, self).__init__()
        self.ln_f = torch.nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.config = config
        
    def forward(self, hidden_states):
        hidden_states = self.ln_f(hidden_states)
        return hidden_states

class GPT2LMHead(torch.nn.Module):
    def __init__(self, config):
        super(GPT2LMHead, self).__init__()
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.config = config

    def forward(self, hidden_states):
        lm_logits = self.lm_head(hidden_states)
        return lm_logits

class GPT2SimpleModel(torch.nn.Module):
    def __init__(self, config, gl_window_size):
        super(GPT2SimpleModel, self).__init__()
        self.embed = GPT2Embeddings(config)
        # self.layer = torch.nn.ModuleList([ GL_GPT2Layer(config) 
        #                                     for _ in range(config.n_layer) ])
        self.layer = Transformer(config, gl_window_size)
        self.ln = GPT2LayerNorm(config)
        self.lm_head = GPT2LMHead(config)
        self.config = config
        # Initialize weights
        self.apply(self._init_weights)
       
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (torch.nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, labels=None):
        hidden_states = self.embed(input_ids)
        # for l in self.layer:
        #     hidden_states = l(hidden_states)   
        for l in self.layer.layers:
            hidden_states = l(hidden_states)   
        hidden_states = self.ln(hidden_states)
        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,)
        if labels is not None: # criterition
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs # (lm_logits,) or (loss, lm_logits)
    
    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        from .file_utils import WEIGHTS_NAME
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))

    # 1.根据给定的模型名称(pretrained_model_name_or_path)获取对应的URL
    # 2.从URL下载模型权重，若已下载，直接返回缓存的文件路径
    # 3.实例化一个模型对象(当前类：GPT2SimpleModel)
    # 4.加载权重文件到cpu上，是一个字典
    # 5.复制权重到新的状态字典中
    # 6.将整理好的新状态字典加载到模型中
    # 7.确保模型的输出embedding权重和输入embedding权重是相同的（权重共享）
    # 8.将模型设置为评估模式，这将禁用 DropOut 模块和其他仅在训练时启用的模块
    # 返回模型
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        assert pretrained_model_name_or_path is not None
        config = kwargs.pop("config", None)
        cache_dir = kwargs.pop("cache_dir", None)
        gl_window_size = kwargs.pop("gl_window_size", None)
        
        # Load config if we don't provide a configuration
        if config is None:
            config_path = pretrained_model_name_or_path
            config, _ = GPT2Config.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=False,
                resume_download=False,
                proxies=None,
                local_files_only=False,
                **kwargs,
            )

        # Load model
        # 1.根据给定的模型名称获取对应的URL
        # 如果 pretrained_model_name_or_path 在 GPT2_PRETRAINED_MODEL_ARCHIVE_MAP 中，则从映射中获取权重文件的路径(URL)
        if pretrained_model_name_or_path in GPT2_PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = GPT2_PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            from .file_utils import WEIGHTS_NAME
            if os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                # Load from a PyTorch checkpoint
                archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            else:
                raise EnvironmentError
        else:
            raise ValueError

        # redirect to the cache, if necessary
        # 2.从URL下载模型权重，若已下载，直接返回缓存的文件路径
        try:
            from .file_utils import cached_path
            # 如果输入是远程 URL，则调用 get_from_cache 方法从缓存中获取文件路径（如果需要则下载）
            resolved_archive_file = cached_path(
                archive_file,                # URL
                cache_dir=cache_dir,
                force_download=False,
                proxies=None,
                resume_download=False,
                local_files_only=False,
            )
        except EnvironmentError:
            if pretrained_model_name_or_path in GPT2_PRETRAINED_MODEL_ARCHIVE_MAP:
                msg = "Couldn't reach server at '{}' to download pretrained weights.".format(archive_file)
            else:
                msg = "Model '{}' was not found."
            raise EnvironmentError(msg)

        if resolved_archive_file == archive_file:
            logger.info("loading weights file {}".format(archive_file))
        else:
            logger.info("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))

        # Instantiate model.
        # 3.实例化一个模型对象（cls就是当前类）
        model = cls(config, gl_window_size)

        for name, module in model.named_modules():
            if name.endswith("language_model"):
                print("language_model!")

        exit(0)

        all_params_in_pinned_memory = True
        for name, param in model.named_parameters():
            print(name)
            if not param.is_pinned():
                all_params_in_pinned_memory = False
                print(f"Parameter {name} is not in pinned memory.")
            else:
                print(f"Parameter {name} is in pinned memory.")

        assert all_params_in_pinned_memory, "Not all parameters are in pinned memory."

        print("All parameters are correctly in pinned memory.")

        exit(0)

        try:
            # 4.加载权重文件到cpu上，是一个字典
            src_state_dict = torch.load(resolved_archive_file, map_location="cpu")
        except Exception:
            raise OSError("Unable to load weights from pytorch checkpoint file. ")
        
        print("======== src state dict ========")
        print(len(src_state_dict))
        for key, val in src_state_dict.items():
            print("{} : {}".format(key, val.shape))
        print("=================================")

        print("======= model state dict =======")
        print(len(model.state_dict()))
        for key, val in model.state_dict().items():
            print("{} : {}".format(key, val.shape))
        print("=================================")

        # Load state dict by order
        # 5.复制权重到新的状态字典中
        from collections import OrderedDict as ODict
        assert isinstance(src_state_dict, ODict)
        src_sdict_iter = iter(src_state_dict.items())
        new_state_dict = ODict()
        # 将源状态字典(src_state_dict)中的参数加载到新的状态字典中，并确保形状和数据类型匹配
        for key, val in model.state_dict().items():
            # print(f"key:{key},value:{val}")
            try:
                src_key, src_val = next(src_sdict_iter)
                assert val.shape == src_val.shape
                assert val.dtype == src_val.dtype
                new_state_dict[key] = src_val
            # 即当源状态字典(src_state_dict)中的参数已经用完时，使用模型的默认参数
            except StopIteration:
                new_state_dict[key] = val
        # 6.将整理好的新状态字典加载到模型中
        model.load_state_dict(new_state_dict)
        logger.info("successfully loaded weights file {}".format(archive_file))
        
        # model.tie_weights()  # make sure token embedding weights are still tied if needed
        # 7.确保模型的输出embedding权重和输入embedding权重是相同的（权重共享）
        model.lm_head.lm_head.weight = model.embed.wte.weight # output_embeddings.weight = input_embeddings.weight

        # Set model in evaluation mode to desactivate DropOut modules by default
        # 8.将模型设置为评估模式，这将禁用 DropOut 模块和其他仅在训练时启用的模块
        model.eval()
        
        return model
    
    @classmethod
    def from_config(cls, config):
        # Instantiate model.
        model = cls(config)
        
        # model.tie_weights()  # make sure token embedding weights are still tied if needed
        model.lm_head.lm_head.weight = model.embed.wte.weight # output_embeddings.weight = input_embeddings.weight

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()
        
        return model
