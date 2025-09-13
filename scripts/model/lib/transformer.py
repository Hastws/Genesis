#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lib.transformer_layer import MultiHeadAttention
from model.lib.transformer_layer import PositionwiseFeedForward

from model.lib.device import best_device


debug = False


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_inner=1024,
        n_layers=3,
        n_head=2,
        d_k=128,
        d_v=128,
        dropout=0.1,
    ):

        super().__init__()

        self.encoder = Encoder(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

    def forward(self, x, mask=None):
        enc_output, *_ = self.encoder(x, mask=mask)

        return enc_output


class Encoder(nn.Module):
    def __init__(
        self,
        n_layers=3,
        n_head=2,
        d_k=128,
        d_v=128,
        d_model=256,
        d_inner=1024,
        dropout=0.1,
    ):

        super().__init__()

        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, mask=None):

        enc_output = x
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, mask=mask)

        del enc_slf_attn

        return (enc_output,)


class EncoderLayer(nn.Module):
    def __init__(
        self, d_model=256, d_inner=1024, n_head=2, d_k=128, d_v=128, dropout=0.1
    ):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.ln1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask=None):
        att_out, enc_slf_attn = self.slf_attn(x, x, x, mask=mask)

        att_out = self.drop1(att_out)
        out_1 = self.ln1(x + att_out)

        ffn_out = self.pos_ffn(out_1)

        ffn_out = self.drop2(ffn_out)
        out = self.ln2(out_1 + ffn_out)

        return out, enc_slf_attn


def test():
    torch.manual_seed(42)

    device = best_device()
    B, L = 2, 16
    d_model = 256

    model = Transformer(
        d_model=d_model,
        d_inner=1024,
        n_layers=3,
        n_head=2,
        d_k=128,
        d_v=128,
        dropout=0.1,
    ).to(device)

    x = torch.randn(B, L, d_model, device=device, requires_grad=True)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    mask = causal.unsqueeze(0).expand(B, -1, -1)

    enc_out = model(x, mask=mask)
    print("enc_out:", enc_out.shape) if debug else None

    loss = enc_out.pow(2).mean()
    loss.backward()
    print("loss:", loss) if debug else None

    total_norm = 0.0
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().float().pow(2).sum().item()
    print("grad_l2_norm:", total_norm**0.5) if debug else None


if __name__ == "__main__":
    test()
