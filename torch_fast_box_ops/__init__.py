"""Pytorch Faster Box Opersations"""

from pathlib import Path

import torch
from torch import Tensor

# Grab binding from goofy-ahh folder above after installation
torch.ops.load_library(
    next(
        str(p)
        for p in Path(__file__).parent.parent.iterdir()
        if p.suffix == ".so" and p.stem.startswith("torch_fast_box_ops")
    )
)


def _box_convert_context(ctx, inputs: tuple[Tensor, str, str], output: Tensor):
    """Save input and output formats for backward pass."""
    _, ctx.in_fmt, ctx.out_fmt = inputs


def _box_convert_backward(ctx, grad: Tensor):
    grad_input = torch.ops.box_ops.box_convert_backward(grad, ctx.in_fmt, ctx.out_fmt)
    return grad_input, None, None


torch.library.register_autograd(
    "box_ops::box_convert", _box_convert_backward, setup_context=_box_convert_context
)


def box_convert(boxes: Tensor, in_fmt: str, out_fmt: str) -> Tensor:
    """
    Convert bounding boxes between different formats.

    Args:
        boxes (Tensor): Bounding boxes in the input format.
        in_fmt (str): Input format ('xyxy', 'xywh', 'cxcywh').
        out_fmt (str): Output format ('xyxy', 'xywh', 'cxcywh').

    Returns:
        Tensor: Bounding boxes in the output format.
    """
    return torch.ops.box_ops.box_convert(boxes, in_fmt, out_fmt)
