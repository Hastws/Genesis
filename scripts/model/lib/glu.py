#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

debug = False


class GLU(nn.Module):

    def __init__(self, input_size=384, context_size=1024, output_size=1024):
        super().__init__()
        self.fc_1 = nn.Linear(context_size, input_size)
        self.fc_2 = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, context):
        gate = self.sigmoid(self.fc_1(context))
        gated_input = gate * x
        output = self.fc_2(gated_input)

        return output


def test():
    context = torch.randn(5, 32)
    x = torch.randn(5, 16)
    output_size = 24

    glu = GLU(
        input_size=x.shape[-1], context_size=context.shape[-1], output_size=output_size
    )
    print("context:", context) if debug else None
    print("context shape:", context.shape) if debug else None
    print("x:", x) if debug else None
    print("x shape:", x.shape) if debug else None
    output = glu(x, context)
    print("output:", output) if debug else None
    print("output shape:", output.shape) if debug else None


if __name__ == "__main__":
    test()
