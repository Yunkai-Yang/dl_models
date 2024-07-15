"""
Some of the components used in Transformer are defined here.
References:
1) nano-GPT released by Karpathy:
https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class LayerNorm(nn.Module):
    # LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        n_embd, bias = config.n_embd, config.bias

        self.mappers = nn.ModuleDict(dict(
            q_m=nn.Linear(n_embd, n_embd, bias=bias),
            k_m=nn.Linear(n_embd, n_embd, bias=bias),
            v_m=nn.Linear(n_embd, n_embd, bias=bias),
        ))
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, q, k, v, mask=None):
        B, T, C = q.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = [mapper(input)
                   .view(B, T, self.n_head, C // self.n_head)
                   .transpose(1, 2) for (mapper, input)
                   in zip(self.mappers.values(), [q, k, v])]

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config, dilatation=4):
        super().__init__()
        n_embd, bias = config.n_embd, config.bias

        self.c_fc = nn.Linear(n_embd, dilatation * n_embd, bias=bias)
        self.act_func = getattr(nn, config.act_func, nn.ReLU)()
        self.c_proj = nn.Linear(dilatation * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act_func(x)
        x = self.c_proj(x)
        return self.dropout(x)


class WordEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.lut = nn.Embedding(config.vocab_size, config.n_embd)
        self.n_embd = config.n_embd

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.n_embd)


class PositionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dropout = nn.Dropout(p=config.dropout)
        n_embd = config.n_embd
        block_size = config.block_size

        pe = torch.zeros(block_size, n_embd)
        position = torch.arange(block_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., n_embd, 2) * -(math.log(10000.0) / n_embd))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class EncoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config=config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, mask=None):
        x = self.ln_1(x)
        x = x + self.attn(x, x, x, mask)  # q,k,v,mask
        x = x + self.mlp(self.ln_2(x))
        return x


class DecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        n_embd, bias = config.n_embd, config.bias
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.cross_attn = CausalSelfAttention(config=config)
        self.t_attn = CausalSelfAttention(config=config)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(config)
        self.ln_3 = LayerNorm(n_embd, bias=bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, enc, target, causal_mask=None, padding_mask=None):
        target = self.ln_1(target)
        q = self.t_attn(target, target, target, mask=causal_mask)
        q = self.dropout(self.ln_2(q + target))
        out = q + self.cross_attn(q, enc, enc, mask=padding_mask)
        return out + self.mlp(self.ln_3(out))