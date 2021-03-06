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


def create_padding_mask(seq):
    """
    Mask all the pad tokens in the batch of sequence. It ensures that the model does not treat padding
    as the input.
    The mask indicates where pad value 0 is present: it outputs a 1 at those locations, and a 0 otherwise.
    The input shape is [batch, seq_len], the output shape: [batch, 1, 1, seq_len]
    """
    # Everything that is zero will become "False" otherwise it will become "True"
    seq_masked = seq == 0
    # add extra dimensions to add the padding
    # to the attention logits.
    seq_masked_unsqueezed = seq_masked.unsqueeze(1)
    return seq_masked_unsqueezed.int()


def create_look_ahead_mask(size):
    """
    Creates a mask that avoid the decoder attention to see anything in the future, allow only past data
    """
    attn_shape = (size, size)
    # Return a copy of an array with the elements below the k-th diagonal zeroed (we want as ones...)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # Invert the elements (now we have our ones bellow the k-th diagonal)
    inverted_values = torch.from_numpy(subsequent_mask) == 0
    return inverted_values.int()


class WordEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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


def scaled_dot_product_attn(query, key, value, mask=None, dropout=None):
    # Input shape(key, query and values): [batch, num_heads, seq_len, depth]
    # Get the depth
    d_k = query.size(-1)

    # This tensor multiplication is actually the place where computation and memory are most intensive
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # Apply mask (used Decoder Attention) to avoid looking into the future tokens (casual)
    if mask is not None:
        scores += (mask * -1e9)
        #scores = scores.masked_fill(mask == 0, -1e9)

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
        scaled_attn, self.attn_weights = scaled_dot_product_attn(
            query=q, key=k, value=v, mask=mask, dropout=self.dropout)

        # Concat the num_heads and depth (back to d_model) into shape [batch_size, seq_len, d_model]
        concat_scaled_attn = scaled_attn.transpose(1, 2).contiguous().view(num_batches, -1, self.num_heads * self.depth)
        output = self.linear_concat(concat_scaled_attn)
        return output, self.attn_weights


class TransformerOutputGenerator(nn.Module):
    """
    Last layer of the decoder where the classes (or words) are generated, don't forget that on the
    transformer decoder/encoder arquitecture the output will be fed back to the decoder one step
    at a time
    """

    def __init__(self, d_model, num_classes_or_vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, num_classes_or_vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.multi_head_attn = MultiHeadedAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.point_wise_ff = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        # Key==Query==Value (Self-attention) shapes: [batch_size, input_seq_len, d_model]
        attn_output, _ = self.multi_head_attn(key=x, query=x, value=x, mask=mask)
        attn_output = self.dropout_1(attn_output)
        output_1 = self.layer_norm_1(x + attn_output)

        ffn_output = self.point_wise_ff(output_1)
        ffn_output = self.dropout_2(ffn_output)
        output_2 = self.layer_norm_1(output_1 + ffn_output)
        # Output shape: [batch_size, input_seq_len, d_model]
        return output_2


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.multi_head_attn_1 = MultiHeadedAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.multi_head_attn_2 = MultiHeadedAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.point_wise_ff = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.dropout_3 = nn.Dropout(p=dropout)

    def forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None):
        # Notice that we have an input x and the encoder (key, values) all with shape:
        # [batch_size, input_seq_len, d_model]
        # x will be the last output sentence (created step by step)
        attn_output_1, attn_weights_1 = self.multi_head_attn_1(
            key=x, query=x, value=x, mask=look_ahead_mask)
        attn_output_1 = self.dropout_1(attn_output_1)
        output_1 = self.layer_norm_1(attn_output_1 + x)

        # Observer that the output of the decoder can query information learned on the encoder
        attn_output_2, attn_weights_2 = self.multi_head_attn_2(
            value=encoder_output, key=encoder_output, query=output_1, mask=padding_mask)
        attn_output_2 = self.dropout_2(attn_output_2)
        output_2 = self.layer_norm_2(attn_output_2 + output_1)

        ffn_output = self.point_wise_ff(output_2)
        ffn_output = self.dropout_3(ffn_output)
        output_3 = self.layer_norm_3(output_2 + ffn_output)
        return output_3, attn_weights_1, attn_weights_2


class TransformerEncoder(nn.Module):
    def __init__(self, n_x, d_model, num_heads, d_ff, input_vocab_size, max_len, rate=0.1):
        super().__init__()
        self.d_model = d_model
        # Nx will be the number of transformer blocks sequentially connected
        self.n_x = n_x
        self.embedding = WordEmbeddings(d_model, input_vocab_size)
        self.positional_encoder = PositionalEncoding(d_model, dropout=rate, max_len=max_len)
        # Create a list of
        self.trf_encoder_blocks = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, rate) for _ in range(n_x)])
        self.dropout = nn.Dropout(p=rate)

    def forward(self, x, mask):
        # After embedding the shape will be [batch_size, input_seq_len, d_model]
        x = self.embedding(x)
        x = self.positional_encoder(x)
        x = self.dropout(x)

        # Run N transformer blocks
        for i in range(self.n_x):
            x = self.trf_encoder_blocks[i](x, mask)

        # Output shape will also be: [batch_size, input_seq_len, d_model]
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, n_x, d_model, num_heads, d_ff, output_vocab_size, max_len, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_x = n_x
        self.embedding = WordEmbeddings(d_model, output_vocab_size)
        self.positional_encoder = PositionalEncoding(d_model, dropout=rate, max_len=max_len)

        self.trf_decoder_blocks = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, rate) for _ in range(n_x)])
        self.dropout = nn.Dropout(p=rate)

    def forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None):
        attention_weights = {}
        # After embedding the shape will be [batch_size, input_seq_len, d_model]
        x = self.embedding(x)
        x = self.positional_encoder(x)
        x = self.dropout(x)

        # Run N transformer blocks
        for i in range(self.n_x):
            x, attn_weight_1, attn_weight_2 = self.trf_decoder_blocks[i](
                x, encoder_output, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i + 1}_attn_weight_1'] = attn_weight_1
            attention_weights[f'decoder_layer{i + 1}_attn_weight_2'] = attn_weight_2

        # Output shape will also be: [batch_size, input_seq_len, d_model]
        return x, attention_weights


class TransformerEncoderDecoder(nn.Module):
    """
    Transformer architecture commonly used on Translation tasks
    """

    def __init__(self, n_x, d_model, num_heads, d_ff,
                 input_vocab_size, output_vocab_size, max_len_input, max_len_output, rate=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(n_x=n_x, d_model=d_model,
                                          num_heads=num_heads, d_ff=d_ff,
                                          input_vocab_size=input_vocab_size,
                                          max_len=max_len_input, rate=rate)

        self.decoder = TransformerDecoder(n_x=n_x, d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                                          output_vocab_size=output_vocab_size,
                                          max_len=max_len_output, rate=rate)
        self.final_layer = nn.Linear(d_model, output_vocab_size)

    @staticmethod
    def create_masks(inp, tar):
        """
        Create masks to avoid calculating values on the PAD token and to
        enforce causality (only take into account past tokens) on the decoder attention
        """
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tar.shape[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = torch.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, look_ahead_mask, dec_padding_mask

    def forward(self, inputs, current_output):
        """
        Remember on the transformers the input is given all at once, while the outputs are
        gathered step-by step
        """
        # Create masks to avoid computing over PAD and decoder to see the future (masked attention)
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inputs, current_output)

        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inputs, enc_padding_mask)

        dec_output, attn_weights = self.decoder(current_output, enc_output, look_ahead_mask, dec_padding_mask)

        next_output = self.final_layer(dec_output)
        return next_output, attn_weights
