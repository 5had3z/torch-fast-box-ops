"""Pytorch Faster Box Operations"""

from pathlib import Path

import torch
from torch import Tensor

torch.ops.load_library(
    next(str(p) for p in Path(__file__).parent.iterdir() if p.suffix == ".so")
)


def _box_convert_context(ctx, inputs: tuple[Tensor, str, str], output: Tensor):
    """Save input and output formats for backward pass."""
    _, ctx.in_fmt, ctx.out_fmt = inputs


def _box_convert_backward(ctx, grad: Tensor):
    grad_input = torch.ops.box_ops.box_convert_backward(
        grad.contiguous(), ctx.in_fmt, ctx.out_fmt
    )
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
    return torch.ops.box_ops.box_convert(boxes.contiguous(), in_fmt, out_fmt)


def _box_area_context(ctx, inputs: tuple[Tensor, str, str], output: Tensor):
    """Save the boxes tensor for backward pass."""
    ctx.save_for_backward(inputs[0])


def _box_area_backward(ctx, grad: Tensor):
    boxes: Tensor
    (boxes,) = ctx.saved_tensors
    grad_input = torch.ops.box_ops.box_area_backward(
        grad.contiguous(), boxes.contiguous()
    )
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
    return torch.ops.box_ops.box_area(boxes.contiguous())


def _loss_inter_union_context(ctx, inputs: tuple[Tensor, Tensor], output: Tensor):
    """Save the boxes tensors for backward pass."""
    ctx.save_for_backward(inputs[0], inputs[1])


def _loss_inter_union_backward(ctx, grad_inter: Tensor, grad_union: Tensor):
    boxes1: Tensor
    boxes2: Tensor
    (boxes1, boxes2) = ctx.saved_tensors
    grad_box1, grad_box2 = torch.ops.box_ops._loss_inter_union_backward(
        grad_inter.contiguous(),
        grad_union.contiguous(),
        boxes1.contiguous(),
        boxes2.contiguous(),
    )
    return grad_box1, grad_box2


torch.library.register_autograd(
    "box_ops::_loss_inter_union",
    _loss_inter_union_backward,
    setup_context=_loss_inter_union_context,
)


def _loss_inter_union(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    """
    Compute intersection and union areas for two sets of boxes.

    Args:
        boxes1 (Tensor): First set of boxes.
        boxes2 (Tensor): Second set of boxes.

    Returns:
        tuple: Intersection and union areas.
    """
    return torch.ops.box_ops._loss_inter_union(boxes1.contiguous(), boxes2.contiguous())


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Compute the Intersection over Union (IoU) for two sets of bounding boxes.

    Args:
        boxes1 (Tensor): First set of M boxes in format [x1, y1, x2, y2].
        boxes2 (Tensor): Second set of N boxes in format [x1, y1, x2, y2].

    Returns:
        Tensor: [M, N] IoU values for each pair of boxes.
    """
    return torch.ops.box_ops.box_iou(boxes1.contiguous(), boxes2.contiguous())


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Compute the Generalized Intersection over Union (GIoU) for two sets of bounding boxes.

    Args:
        boxes1 (Tensor): First set of M boxes in format [x1, y1, x2, y2].
        boxes2 (Tensor): Second set of N boxes in format [x1, y1, x2, y2].

    Returns:
        Tensor: [M, N] GIoU values for each pair of boxes.
    """
    return torch.ops.box_ops.generalized_box_iou(
        boxes1.contiguous(), boxes2.contiguous()
    )


def distance_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute the Complete Intersection over Union (CIoU) for two sets of bounding boxes.

    Args:
        boxes1 (Tensor): First set of M boxes in format [x1, y1, x2, y2].
        boxes2 (Tensor): Second set of N boxes in format [x1, y1, x2, y2].

    Returns:
        Tensor: [M, N] CIoU values for each pair of boxes.
    """
    return torch.ops.box_ops.distance_box_iou(boxes1.contiguous(), boxes2.contiguous())


def _box_iou_loss_context(
    ctx, inputs: tuple[Tensor, Tensor, str, float], output: Tensor
):
    """Save the boxes tensors for backward pass."""
    ctx.save_for_backward(inputs[0], inputs[1])
    ctx.eps = inputs[2]


def _generalized_box_iou_loss_backward(ctx, grad: Tensor):
    boxes1: Tensor
    boxes2: Tensor
    (boxes1, boxes2) = ctx.saved_tensors
    grad_boxes1, grad_boxes2 = torch.ops.box_ops.generalized_box_iou_loss_backward(
        grad.contiguous(), boxes1.contiguous(), boxes2.contiguous(), ctx.eps
    )
    return grad_boxes1, grad_boxes2, None


torch.library.register_autograd(
    "box_ops::generalized_box_iou_loss",
    _generalized_box_iou_loss_backward,
    setup_context=_box_iou_loss_context,
)


def generalized_box_iou_loss(
    boxes1: Tensor, boxes2: Tensor, reduction: str = "none", eps: float = 1e-7
) -> Tensor:
    """
    Compute the Generalized IoU loss for two sets of bounding boxes.

    Args:
        boxes1 (Tensor): First set of boxes in format [x1, y1, x2, y2].
        boxes2 (Tensor): Second set of boxes in format [x1, y1, x2, y2].
        reduction (str): Reduction method ('none', 'mean', 'sum').
        eps (float): Small value to prevent division by zero.

    Returns:
        Tensor: Loss values with the specified reduction applied.
    """
    loss: Tensor = torch.ops.box_ops.generalized_box_iou_loss(
        boxes1.contiguous(), boxes2.contiguous(), eps
    )

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n "
            "Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


def _distance_box_iou_loss_backward(ctx, grad: Tensor):
    boxes1: Tensor
    boxes2: Tensor
    (boxes1, boxes2) = ctx.saved_tensors
    grad_boxes1, grad_boxes2 = torch.ops.box_ops.distance_box_iou_loss_backward(
        grad.contiguous(), boxes1.contiguous(), boxes2.contiguous(), ctx.eps
    )
    return grad_boxes1, grad_boxes2, None


torch.library.register_autograd(
    "box_ops::distance_box_iou_loss",
    _distance_box_iou_loss_backward,
    setup_context=_box_iou_loss_context,
)


def distance_box_iou_loss(
    boxes1: Tensor, boxes2: Tensor, reduction: str = "none", eps: float = 1e-7
) -> Tensor:
    """
    Compute the Distance IoU loss for two sets of bounding boxes.

    Args:
        boxes1 (Tensor): First set of boxes in format [x1, y1, x2, y2].
        boxes2 (Tensor): Second set of boxes in format [x1, y1, x2, y2].
        reduction (str): Reduction method ('none', 'mean', 'sum').
        eps (float): Small value to prevent division by zero.

    Returns:
        Tensor: Loss values with the specified reduction applied.
    """
    loss: Tensor = torch.ops.box_ops.distance_box_iou_loss(
        boxes1.contiguous(), boxes2.contiguous(), eps
    )

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n "
            "Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss
