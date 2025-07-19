"""Pytorch Faster Box Operations"""

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


def _box_area_context(ctx, inputs: tuple[Tensor, str, str], output: Tensor):
    """Save the boxes tensor for backward pass."""
    ctx.save_for_backward(inputs[0])


def _box_area_backward(ctx, grad: Tensor):
    (boxes,) = ctx.saved_tensors
    grad_input = torch.ops.box_ops.box_area_backward(grad, boxes)
    return grad_input


torch.library.register_autograd(
    "box_ops::box_area", _box_area_backward, setup_context=_box_area_context
)


def box_area(boxes: Tensor) -> Tensor:
    """
    Compute the area of bounding boxes.

    Args:
        boxes (Tensor): Bounding boxes in the format [x1, y1, x2, y2].

    Returns:
        Tensor: Areas of the bounding boxes.
    """
    return torch.ops.box_ops.box_area(boxes)
