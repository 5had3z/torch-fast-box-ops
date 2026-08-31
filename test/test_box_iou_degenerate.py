"""Numerical robustness of the IoU losses on degenerate (zero-area) boxes.

A zero-area box is easy to produce in a real detector: a DFL head whose distribution
collapses onto the first bin emits ``x1 == x2``, and mosaic/affine augmentation can clip
a ground truth box down to nothing. If the loss or its gradient turns non-finite there,
the NaN spreads across the whole batch on the backward pass and the AMP grad scaler
silently drops the step, so these cases are worth pinning down explicitly.

Comparison against torchvision
------------------------------
``torchvision`` is the reference for the normal cases, but it is *not* finite on all of
the degenerate ones, so equality is only asserted where torchvision itself produces a
finite result. Measured against torchvision 0.28.0:

* ``generalized_box_iou_loss`` and ``distance_box_iou_loss`` are finite everywhere, in
  both float32 and float16.
* ``complete_box_iou_loss`` returns NaN for *any* zero-area box, in the forward as well
  as the backward. It computes ``atan(w / h)`` with no epsilon on the denominator
  (``torchvision/ops/ciou_loss.py``), so a zero-height box gives ``0 / 0``. tfbo guards
  that divide and stays finite, hence the skips below rather than a strict comparison.
"""

from typing import Callable

import pytest
import torch
from torchvision.ops.ciou_loss import complete_box_iou_loss as tv_complete_box_iou_loss
from torchvision.ops.diou_loss import distance_box_iou_loss as tv_distance_box_iou_loss
from torchvision.ops.giou_loss import (
    generalized_box_iou_loss as tv_generalized_box_iou_loss,
)

from torch_fast_box_ops import complete_box_iou_loss as tfbo_complete_box_iou_loss
from torch_fast_box_ops import distance_box_iou_loss as tfbo_distance_box_iou_loss
from torch_fast_box_ops import generalized_box_iou_loss as tfbo_generalized_box_iou_loss

LossFn = Callable[..., torch.Tensor]

# name -> (tfbo fn, torchvision fn)
LOSS_FNS: dict[str, tuple[LossFn, LossFn]] = {
    "giou": (tfbo_generalized_box_iou_loss, tv_generalized_box_iou_loss),
    "diou": (tfbo_distance_box_iou_loss, tv_distance_box_iou_loss),
    "ciou": (tfbo_complete_box_iou_loss, tv_complete_box_iou_loss),
}

# name -> (box1, box2) in xyxy
DEGENERATE_CASES: dict[str, tuple[list[float], list[float]]] = {
    "normal": ([10.0, 10.0, 30.0, 40.0], [12.0, 11.0, 33.0, 39.0]),
    "zero_width_box1": ([10.0, 10.0, 10.0, 40.0], [12.0, 11.0, 33.0, 39.0]),
    "zero_height_box1": ([10.0, 10.0, 30.0, 10.0], [12.0, 11.0, 33.0, 39.0]),
    "zero_width_box2": ([10.0, 10.0, 30.0, 40.0], [12.0, 11.0, 12.0, 39.0]),
    "zero_height_box2": ([10.0, 10.0, 30.0, 40.0], [12.0, 11.0, 33.0, 11.0]),
    "point_box1": ([10.0, 10.0, 10.0, 10.0], [12.0, 11.0, 33.0, 39.0]),
    "point_box2": ([10.0, 10.0, 30.0, 40.0], [12.0, 12.0, 12.0, 12.0]),
    "both_points": ([10.0, 10.0, 10.0, 10.0], [12.0, 12.0, 12.0, 12.0]),
    "both_points_coincident": ([10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0]),
    "identical": ([10.0, 10.0, 30.0, 40.0], [10.0, 10.0, 30.0, 40.0]),
    "disjoint": ([0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0]),
    "tiny_subpixel": ([10.0, 10.0, 10.001, 10.001], [10.0, 10.0, 10.002, 10.002]),
}


def _forward_backward(
    loss_fn: LossFn,
    box1: list[float],
    box2: list[float],
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a loss over a single box pair, returning (loss, grad_box1, grad_box2)."""
    boxes1 = torch.tensor([box1], dtype=dtype, device=device, requires_grad=True)
    boxes2 = torch.tensor([box2], dtype=dtype, device=device, requires_grad=True)
    loss = loss_fn(boxes1, boxes2, reduction="none")
    loss.sum().backward()
    assert boxes1.grad is not None and boxes2.grad is not None
    return loss.detach(), boxes1.grad, boxes2.grad


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
@pytest.mark.parametrize("loss_name", list(LOSS_FNS))
@pytest.mark.parametrize("case_name", list(DEGENERATE_CASES))
def test_degenerate_boxes_are_finite(
    device: str, dtype: torch.dtype, loss_name: str, case_name: str
):
    """No degenerate box may produce a non-finite loss or gradient."""
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 is not supported on cpu")

    tfbo_fn, _ = LOSS_FNS[loss_name]
    box1, box2 = DEGENERATE_CASES[case_name]
    loss, grad1, grad2 = _forward_backward(tfbo_fn, box1, box2, dtype, device)

    assert torch.isfinite(loss).all(), f"non-finite loss {loss.tolist()}"
    assert torch.isfinite(grad1).all(), f"non-finite grad_box1 {grad1.tolist()}"
    assert torch.isfinite(grad2).all(), f"non-finite grad_box2 {grad2.tolist()}"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("loss_name", list(LOSS_FNS))
@pytest.mark.parametrize("case_name", list(DEGENERATE_CASES))
def test_degenerate_boxes_match_torchvision(device: str, loss_name: str, case_name: str):
    """Agree with torchvision on every degenerate case torchvision handles itself.

    See the module docstring: torchvision's ciou is non-finite for any zero-area box, so
    those combinations skip rather than compare.
    """
    tfbo_fn, tv_fn = LOSS_FNS[loss_name]
    box1, box2 = DEGENERATE_CASES[case_name]

    tv_loss, tv_grad1, tv_grad2 = _forward_backward(
        tv_fn, box1, box2, torch.float32, device
    )
    if not all(torch.isfinite(t).all() for t in (tv_loss, tv_grad1, tv_grad2)):
        pytest.skip(f"torchvision {loss_name} is non-finite for '{case_name}'")

    tfbo_loss, tfbo_grad1, tfbo_grad2 = _forward_backward(
        tfbo_fn, box1, box2, torch.float32, device
    )
    torch.testing.assert_close(tfbo_loss, tv_loss, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(tfbo_grad1, tv_grad1, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(tfbo_grad2, tv_grad2, rtol=1e-5, atol=2e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("loss_name", list(LOSS_FNS))
def test_degenerate_box_does_not_poison_batch(
    device: str, dtype: torch.dtype, loss_name: str
):
    """One bad box in a batch must not NaN the gradients of every other box.

    This is the failure that matters in training: the losses are summed before
    ``backward``, so a single non-finite entry propagates to every box in the batch and
    the AMP grad scaler drops the whole step.
    """
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 is not supported on cpu")

    tfbo_fn, _ = LOSS_FNS[loss_name]
    good1 = [[10.0, 10.0, 30.0, 40.0], [5.0, 5.0, 25.0, 25.0], [0.0, 0.0, 8.0, 9.0]]
    good2 = [[12.0, 11.0, 33.0, 39.0], [6.0, 4.0, 24.0, 27.0], [1.0, 1.0, 7.0, 8.0]]
    # index 1 of each batch is degenerate
    bad_idx = 1
    boxes1 = torch.tensor(good1, dtype=dtype, device=device)
    boxes2 = torch.tensor(good2, dtype=dtype, device=device)
    boxes1[bad_idx] = torch.tensor([7.0, 7.0, 7.0, 7.0], dtype=dtype, device=device)
    boxes1.requires_grad_(True)
    boxes2.requires_grad_(True)

    loss = tfbo_fn(boxes1, boxes2, reduction="none")
    loss.sum().backward()
    assert boxes1.grad is not None and boxes2.grad is not None

    assert torch.isfinite(loss).all(), f"non-finite loss {loss.tolist()}"
    assert torch.isfinite(boxes1.grad).all(), f"non-finite grad {boxes1.grad.tolist()}"
    assert torch.isfinite(boxes2.grad).all(), f"non-finite grad {boxes2.grad.tolist()}"

    # The healthy rows must be untouched by their degenerate neighbour
    keep = [i for i in range(len(good1)) if i != bad_idx]
    ref1 = torch.tensor([good1[i] for i in keep], dtype=dtype, device=device)
    ref2 = torch.tensor([good2[i] for i in keep], dtype=dtype, device=device)
    ref1.requires_grad_(True)
    ref2.requires_grad_(True)
    ref_loss = tfbo_fn(ref1, ref2, reduction="none")
    ref_loss.sum().backward()

    rtol, atol = (2e-3, 2e-5) if dtype == torch.float16 else (1e-5, 2e-5)
    torch.testing.assert_close(loss[keep], ref_loss, rtol=rtol, atol=atol)
    torch.testing.assert_close(boxes1.grad[keep], ref1.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(boxes2.grad[keep], ref2.grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("loss_name", list(LOSS_FNS))
def test_zero_area_boxes_at_scale(device: str, loss_name: str):
    """A large batch of randomly zeroed-width/height boxes stays finite."""
    tfbo_fn, _ = LOSS_FNS[loss_name]
    generator = torch.Generator(device="cpu").manual_seed(0)
    num_boxes = 4096

    xy = torch.rand(num_boxes, 2, generator=generator) * 100
    wh = torch.rand(num_boxes, 2, generator=generator) * 50
    boxes1 = torch.cat([xy, xy + wh], dim=-1)
    boxes2 = boxes1.roll(1, dims=0).clone()

    # Collapse a third of the boxes in width, height or both
    collapse = torch.randint(0, 3, (num_boxes,), generator=generator)
    boxes1[collapse == 0, 2] = boxes1[collapse == 0, 0]
    boxes1[collapse == 1, 3] = boxes1[collapse == 1, 1]
    boxes2[collapse == 2, 2:] = boxes2[collapse == 2, :2]

    boxes1 = boxes1.to(device).requires_grad_(True)
    boxes2 = boxes2.to(device).requires_grad_(True)

    loss = tfbo_fn(boxes1, boxes2, reduction="none")
    loss.sum().backward()
    assert boxes1.grad is not None and boxes2.grad is not None

    assert torch.isfinite(loss).all(), f"{(~torch.isfinite(loss)).sum()} non-finite"
    assert torch.isfinite(boxes1.grad).all()
    assert torch.isfinite(boxes2.grad).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("loss_name", list(LOSS_FNS))
def test_identical_boxes_are_zero_loss(device: str, loss_name: str):
    """Identical non-degenerate boxes give exactly zero loss and zero gradient."""
    tfbo_fn, _ = LOSS_FNS[loss_name]
    box = [10.0, 10.0, 30.0, 40.0]
    loss, grad1, grad2 = _forward_backward(tfbo_fn, box, box, torch.float32, device)

    torch.testing.assert_close(loss, torch.zeros_like(loss), rtol=0, atol=1e-6)
    torch.testing.assert_close(grad1, torch.zeros_like(grad1), rtol=0, atol=1e-6)
    torch.testing.assert_close(grad2, torch.zeros_like(grad2), rtol=0, atol=1e-6)
