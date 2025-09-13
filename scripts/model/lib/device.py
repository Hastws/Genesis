#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

debug = False


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps"):
        try:
            if torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


def test():
    device = best_device()
    print("best device:", device) if debug else None


if __name__ == "__main__":
    test()
