from torchvision.ops.boxes import box_convert as tv_box_convert
from torch_fast_box_ops import box_convert as tfbo_box_convert

import torch
import pytest


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.int32]
)
@pytest.mark.parametrize("in_fmt", ["xyxy", "xywh", "cxcywh"])
@pytest.mark.parametrize("out_fmt", ["xyxy", "xywh", "cxcywh"])
def test_box_convert(in_fmt: str, out_fmt: str, dtype: torch.dtype):
    torch.manual_seed(0)
    boxes = torch.rand(10, 4) * 100
    boxes = boxes.to(dtype)
    converted = tfbo_box_convert(boxes, in_fmt, out_fmt)
    expected = tv_box_convert(boxes, in_fmt, out_fmt).to(dtype)
    atol = 1e-1 if dtype == torch.float16 else 1e-8
    assert torch.allclose(
        converted, expected, atol=atol
    ), f"Failed for {in_fmt} to {out_fmt}: {converted} != {expected}"
