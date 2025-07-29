import pytest
import torch
from torch.nn import functional as F
from torchvision.ops.boxes import (
    box_area as tv_box_area,
    box_iou as tv_box_iou,
    generalized_box_iou as tv_generalized_box_iou,
)
from torchvision.ops._utils import _loss_inter_union as tv_loss_inter_union
from torch_fast_box_ops import (
    box_area as tfbo_box_area,
    box_iou as tfbo_box_iou,
    _loss_inter_union as tfbo_loss_inter_union,
    generalized_box_iou as tfbo_generalized_box_iou,
)

from utils import make_random_boxes


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_box_area(device: str, dtype: torch.dtype):
    boxes = make_random_boxes("xyxy", 10, dtype=dtype, device=device, normalized=True)
    tv_area = tv_box_area(boxes).to(dtype=dtype)
    tfbo_area = tfbo_box_area(boxes)

    torch.testing.assert_close(tfbo_area, tv_area[..., None])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_box_area_backward(device: str, dtype: torch.dtype):
    tfbo_boxes = make_random_boxes(
        "xyxy", 10, dtype=dtype, device=device, normalized=True
    )
    tv_boxes = tfbo_boxes.clone()
    tv_boxes.requires_grad = True
    tfbo_boxes.requires_grad = True

    # Forward pass
    tv_area = tv_box_area(tv_boxes).to(dtype=dtype)
    tfbo_area = tfbo_box_area(tfbo_boxes)[..., 0]

    with torch.no_grad():
        random_targets = torch.normal(tv_area, std=0.5).to(dtype=dtype, device=device)

    # Backward pass
    F.mse_loss(tv_area, random_targets).backward()
    F.mse_loss(tfbo_area, random_targets).backward()

    # Check gradients
    torch.testing.assert_close(tfbo_boxes.grad, tv_boxes.grad)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("num_batch", [1, 4])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.int32]
)
@pytest.mark.parametrize("num_boxes", [(10, 12), (31, 32), (91, 318)])
def test_box_iou(device: str, num_batch: int, dtype: torch.dtype, num_boxes: tuple):
    boxes1 = make_random_boxes(
        "xyxy", num_boxes[0], dtype=dtype, device=device, num_batch=num_batch, seed=0
    )
    boxes2 = make_random_boxes(
        "xyxy", num_boxes[1], dtype=dtype, device=device, num_batch=num_batch, seed=1
    )

    if num_batch > 1:
        tv_iou = torch.stack([tv_box_iou(b1, b2) for b1, b2 in zip(boxes1, boxes2)])
    else:
        tv_iou = tv_box_iou(boxes1, boxes2)

    tfbo_iou = tfbo_box_iou(boxes1, boxes2)

    if dtype == torch.float16:
        # Torchvision's box_iou has issues with float16 precision
        tv_iou = tv_iou.to(dtype=dtype)
        torch.testing.assert_close(tfbo_iou, tv_iou, rtol=5e-3, atol=5e-5)
    elif dtype == torch.int32:
        torch.testing.assert_close(tfbo_iou, tv_iou, equal_nan=True)
    else:
        torch.testing.assert_close(tfbo_iou, tv_iou)


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("num_batch", [1, 4])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.int32]
)
@pytest.mark.parametrize("num_boxes", [(10, 12), (31, 32), (91, 318)])
def test_box_giou(device: str, num_batch: int, dtype: torch.dtype, num_boxes: tuple):
    boxes1 = make_random_boxes(
        "xyxy", num_boxes[0], dtype=dtype, device=device, num_batch=num_batch, seed=0
    )
    boxes2 = make_random_boxes(
        "xyxy", num_boxes[1], dtype=dtype, device=device, num_batch=num_batch, seed=1
    )

    if num_batch > 1:
        tv_iou = torch.stack(
            [tv_generalized_box_iou(b1, b2) for b1, b2 in zip(boxes1, boxes2)]
        )
    else:
        tv_iou = tv_generalized_box_iou(boxes1, boxes2)

    tfbo_iou = tfbo_generalized_box_iou(boxes1, boxes2)

    if dtype == torch.float16:
        # Torchvision's box_iou has issues with float16 precision
        tv_iou = tv_iou.to(dtype=dtype)
        torch.testing.assert_close(tfbo_iou, tv_iou, rtol=1e-1, atol=5e-3)
    elif dtype == torch.int32:
        torch.testing.assert_close(tfbo_iou, tv_iou, equal_nan=True)
    else:
        torch.testing.assert_close(tfbo_iou, tv_iou)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_loss_inter_union(device: str):
    boxes1 = make_random_boxes(
        "xyxy", 10, dtype=torch.float32, device=device, normalized=True
    )
    boxes2 = make_random_boxes(
        "xyxy", 10, dtype=torch.float32, device=device, normalized=True
    )

    tv_inter, tv_union = tv_loss_inter_union(boxes1, boxes2)
    tfbo_inter, tfbo_union = tfbo_loss_inter_union(boxes1, boxes2)

    torch.testing.assert_close(tfbo_inter, tv_inter)
    torch.testing.assert_close(tfbo_union, tv_union)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_loss_inter_union_backward(device: str, dtype: torch.dtype):
    boxes1_tfbo = make_random_boxes(
        "xyxy", 10, dtype=dtype, device=device, normalized=True, seed=0
    )
    boxes2_tfbo = make_random_boxes(
        "xyxy", 10, dtype=dtype, device=device, normalized=True, seed=1
    )

    boxes1_tv = boxes1_tfbo.clone()
    boxes2_tv = boxes2_tfbo.clone()

    boxes1_tfbo.requires_grad = True
    boxes2_tfbo.requires_grad = True
    boxes1_tv.requires_grad = True
    boxes2_tv.requires_grad = True

    tv_inter, tv_union = tv_loss_inter_union(boxes1_tv, boxes2_tv)
    tfbo_inter, tfbo_union = tfbo_loss_inter_union(boxes1_tfbo, boxes2_tfbo)
    torch.testing.assert_close(tfbo_inter, tv_inter)
    torch.testing.assert_close(tfbo_union, tv_union)

    # Create random gradients for backward pass
    (1 - tv_inter / tv_union).sum().backward()
    (1 - tfbo_inter / tfbo_union).sum().backward()

    # Check gradients
    torch.testing.assert_close(boxes1_tfbo.grad, boxes1_tv.grad)
    torch.testing.assert_close(boxes2_tfbo.grad, boxes2_tv.grad)
