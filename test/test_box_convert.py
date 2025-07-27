import pytest
import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.ops.boxes import box_convert as tv_box_convert
from torchvision.ops import generalized_box_iou_loss, complete_box_iou_loss

from torch_fast_box_ops import box_convert as tfbo_box_convert


from utils import make_random_boxes


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
@pytest.mark.parametrize("in_fmt", ["xyxy", "xywh", "cxcywh"])
@pytest.mark.parametrize("out_fmt", ["xyxy", "xywh", "cxcywh"])
def test_box_convert(device: str, in_fmt: str, out_fmt: str, dtype: torch.dtype):
    """Test box conversion for various formats and data types, both forward and backward."""
    tfo_boxes = make_random_boxes(in_fmt, 100, dtype, device, normalized=False)
    tv_boxes = tfo_boxes.clone()

    tfo_boxes.requires_grad = True
    tv_boxes.requires_grad = True

    converted = tfbo_box_convert(tfo_boxes, in_fmt, out_fmt)
    expected = tv_box_convert(tfo_boxes, in_fmt, out_fmt).to(dtype)

    scales = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=dtype)
    fake_target = make_random_boxes(out_fmt, 100, dtype, device) * scales

    F.mse_loss(tv_boxes, fake_target).backward()
    F.mse_loss(tfo_boxes, fake_target).backward()

    # Skip values check for cxcywh/xywh because torchvision's intermediate transforms
    # cause slight inaccuracies in the values and gradients at lower precisions.
    if dtype == torch.float16 and {in_fmt, out_fmt} == {"xywh", "cxcywh"}:
        return

    torch.testing.assert_close(converted, expected)
    torch.testing.assert_close(tfo_boxes.grad, tv_boxes.grad)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("in_fmt", ["xyxy", "xywh", "cxcywh"])
@pytest.mark.parametrize("grad_fn", [generalized_box_iou_loss, complete_box_iou_loss])
def test_box_convert_gradients(device: str, in_fmt: str, grad_fn):
    """Test gradients for box conversion using different loss functions."""
    tfo_boxes = make_random_boxes(in_fmt, 100, torch.float32, device)
    tv_boxes = tfo_boxes.clone()

    tfo_boxes.requires_grad = True
    tv_boxes.requires_grad = True

    converted_tfo = tfbo_box_convert(tfo_boxes, in_fmt, "xyxy")
    converted_tv = tv_box_convert(tv_boxes, in_fmt, "xyxy")

    random_target = make_random_boxes("xyxy", 100, torch.float32, device)

    loss_tfo: Tensor = grad_fn(converted_tfo, random_target, reduction="mean")
    loss_tv: Tensor = grad_fn(converted_tv, random_target, reduction="mean")

    torch.testing.assert_close(loss_tfo, loss_tv)

    loss_tfo.backward()
    loss_tv.backward()

    assert tfo_boxes.grad is not None, "TFO boxes gradient is None"
    assert tv_boxes.grad is not None, "TV boxes gradient is None"

    torch.testing.assert_close(tfo_boxes.grad, tv_boxes.grad)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("in_fmt", ["xyxy", "xywh", "cxcywh"])
@pytest.mark.parametrize("out_fmt", ["xyxy", "xywh", "cxcywh"])
def test_box_convert_int32(device: str, in_fmt: str, out_fmt: str):
    """Test box conversion with integer types"""
    dtype = torch.int32
    random_boxes = make_random_boxes(in_fmt, 100, dtype, device)

    converted = tfbo_box_convert(random_boxes, in_fmt, out_fmt)
    expected = tv_box_convert(random_boxes, in_fmt, out_fmt).to(dtype)

    torch.testing.assert_close(converted, expected)
