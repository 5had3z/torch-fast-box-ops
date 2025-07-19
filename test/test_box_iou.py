import pytest
import torch
from torch.nn import functional as F
from torchvision.ops.boxes import box_area as tv_box_area, box_iou as tv_box_iou
from torch_fast_box_ops import box_area as tfbo_box_area, box_iou as tfbo_box_iou

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
    tfo_boxes = make_random_boxes(
        "xyxy", 10, dtype=dtype, device=device, normalized=True
    )
    tv_boxes = tfo_boxes.clone()

    tfo_boxes.requires_grad = True
    tfo_grad = torch.empty_like(tfo_boxes)
    tfo_boxes.register_hook(tfo_grad.copy_)

    tv_boxes.requires_grad = True
    tv_grad = torch.empty_like(tv_boxes)
    tv_boxes.register_hook(tv_grad.copy_)

    # Forward pass
    tv_area = tv_box_area(tv_boxes).to(dtype=dtype)
    tfbo_area = tfbo_box_area(tfo_boxes)[..., 0]

    with torch.no_grad():
        random_targets = torch.normal(tv_area, std=0.5).to(dtype=dtype, device=device)

    # Backward pass
    F.mse_loss(tv_area, random_targets).backward()
    F.mse_loss(tfbo_area, random_targets).backward()

    # Check gradients
    torch.testing.assert_close(tfo_grad, tv_grad)


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("num_batch", [1, 4])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.int32]
)
def test_box_iou(device: str, num_batch: int, dtype: torch.dtype):
    boxes1 = make_random_boxes(
        "xyxy", 10, dtype=dtype, device=device, num_batch=num_batch
    )
    boxes2 = make_random_boxes(
        "xyxy", 12, dtype=dtype, device=device, num_batch=num_batch
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
    else:
        torch.testing.assert_close(tfbo_iou, tv_iou)
