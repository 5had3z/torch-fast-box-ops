import pytest
import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.ops.boxes import box_convert as tv_box_convert
from torchvision.ops import generalized_box_iou_loss, complete_box_iou_loss

from torch_fast_box_ops import box_convert as tfbo_box_convert


from utils import get_atol, make_random_boxes


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
@pytest.mark.parametrize("in_fmt", ["xyxy", "xywh", "cxcywh"])
@pytest.mark.parametrize("out_fmt", ["xyxy", "xywh", "cxcywh"])
def test_box_convert(device: str, in_fmt: str, out_fmt: str, dtype: torch.dtype):
    """Test box conversion for various formats and data types, both forward and backward."""
    tfo_boxes = make_random_boxes(in_fmt, 100, dtype, device, normalized=False)
    tv_boxes = tfo_boxes.clone()

    tfo_boxes.requires_grad = True
    tfo_grad = torch.empty_like(tfo_boxes)
    tfo_boxes.register_hook(tfo_grad.copy_)

    tv_boxes.requires_grad = True
    tv_grad = torch.empty_like(tv_boxes)
    tv_boxes.register_hook(tv_grad.copy_)

    converted = tfbo_box_convert(tfo_boxes, in_fmt, out_fmt)
    expected = tv_box_convert(tfo_boxes, in_fmt, out_fmt).to(dtype)

    assert torch.allclose(
        converted, expected, atol=get_atol(dtype)
    ), f"Conversion Failed {converted} != {expected}"

    scales = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=dtype)
    fake_target = make_random_boxes(out_fmt, 100, dtype, device) * scales

    F.mse_loss(tv_boxes, fake_target).backward()
    F.mse_loss(tfo_boxes, fake_target).backward()

    assert torch.allclose(
        tfo_grad, tv_grad, atol=get_atol(dtype)
    ), f"Backward Failed {tfo_grad} != {tv_grad}"


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

    assert torch.allclose(
        loss_tfo, loss_tv
    ), f"Losses do not match: {loss_tfo} != {loss_tv}"

    loss_tfo.backward()
    loss_tv.backward()

    assert tfo_boxes.grad is not None, "TFO boxes gradient is None"
    assert tv_boxes.grad is not None, "TV boxes gradient is None"

    assert torch.allclose(
        tfo_boxes.grad, tv_boxes.grad
    ), f"Gradients do not match: {tfo_boxes.grad} != {tv_boxes.grad}"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("in_fmt", ["xyxy", "xywh", "cxcywh"])
@pytest.mark.parametrize("out_fmt", ["xyxy", "xywh", "cxcywh"])
def test_box_convert_int32(device: str, in_fmt: str, out_fmt: str):
    """Test box conversion with integer types"""
    dtype = torch.int32
    random_boxes = make_random_boxes(in_fmt, 100, dtype, device)

    converted = tfbo_box_convert(random_boxes, in_fmt, out_fmt)
    expected = tv_box_convert(random_boxes, in_fmt, out_fmt).to(dtype)

    assert torch.allclose(
        converted, expected
    ), f"Conversion Failed {converted} != {expected}"
