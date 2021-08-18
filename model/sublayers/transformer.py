"""
Implement layers used on the transformer architecture direclty 

https://nlp.seas.harvard.edu/2018/04/03/attention.html
https://medium.com/analytics-vidhya/implementing-transformer-from-scratch-in-pytorch-8c872a3044c9
https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
https://hyugen-ai.medium.com/transformers-in-pytorch-from-scratch-for-nlp-beginners-ff3b3d922ef7
https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/10
https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
https://data-science-blog.com/blog/2021/04/07/multi-head-attention-mechanism/
https://theaisummer.com/self-attention/
https://www.tensorflow.org/text/tutorials/transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class WordEmbeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # Buffers won't participate on the optimizer phase
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    # Input shape(key, query and values): [batch, num_heads, seq_len, depth]
    # Get the depth
    d_k = query.size(-1)

    # This tensor multiplication is actually the place where computation and memory are most intensive
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # Apply mask (used Decoder Attention) to avoid looking into the future tokens (casual)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # The attention tensor is used more for debugging
    attn_weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # Scaled attention: [batch, num_heads, seq_len, depth]
    # Attention weights: [batch_size, num_heads, seq_len_q, seq_len_k]
    return torch.matmul(attn_weights, value), attn_weights


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.num_heads = num_heads
        # Instantiate the linear layers for Key, Query, Value
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_concat = nn.Linear(d_model, d_model)
        self.attn_weights = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        num_batches = query.size(0)

        # Apply linear layers to Key,Query,Value their input shape: [batch, seq_len, d_model]
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        # Split d_model into [batch_size, num_heads, seq_len, depth]
        q = q.view(num_batches, -1, self.num_heads, self.depth).transpose(1, 2)
        k = k.view(num_batches, -1, self.num_heads, self.depth).transpose(1, 2)
        v = v.view(num_batches, -1, self.num_heads, self.depth).transpose(1, 2)

        # Compute the scaled attention: [batch_size, num_heads, seq_len, depth] and
        # Attention weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        scaled_attn, self.attn_weights = attention(query, key, value, mask=mask, dropout=self.dropout)

        # Concat the num_heads and depth (back to d_model) into shape [batch_size, seq_len, d_model]
        concat_scaled_attn = scaled_attn.transpose(1, 2).contiguous().view(num_batches, -1, self.num_heads * self.depth)
        output = self.linear_concat(concat_scaled_attn)
        return output


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers"
    """

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
