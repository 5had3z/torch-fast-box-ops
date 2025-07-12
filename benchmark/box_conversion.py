"""Benchmark script for box conversion operations."""

import timeit
from dataclasses import dataclass

import pandas as pd
import torch
from torchvision.ops.boxes import box_convert as tv_box_convert

from torch_fast_box_ops import box_convert as tfbo_box_convert


@dataclass
class BoxResult:
    device: str
    dtype: torch.dtype
    in_fmt: str
    out_fmt: str
    time_tfbo: float
    time_tv: float


def benchmark_box_conversion(
    device: str, dtype: torch.dtype, in_fmt: str, out_fmt: str, num_boxes: int = 10000
):
    """
    Benchmark the box conversion operation.

    Args:
        device (str): Device to run the benchmark on ('cpu' or 'cuda').
        dtype (torch.dtype): Data type of the bounding boxes.
        in_fmt (str): Input format ('xyxy', 'xywh', 'cxcywh').
        out_fmt (str): Output format ('xyxy', 'xywh', 'cxcywh').
    """
    torch.manual_seed(0)
    boxes = torch.rand(num_boxes, 4) * 100
    boxes = boxes.to(dtype=dtype, device=device)

    # Warm-up
    _ = tfbo_box_convert(boxes, in_fmt, out_fmt)

    # Benchmark
    tfbo = timeit.timeit(lambda: tfbo_box_convert(boxes, in_fmt, out_fmt), number=1000)
    tv = timeit.timeit(lambda: tv_box_convert(boxes, in_fmt, out_fmt), number=1000)

    return BoxResult(
        device=device,
        dtype=dtype,
        in_fmt=in_fmt,
        out_fmt=out_fmt,
        time_tfbo=tfbo,
        time_tv=tv,
    )


def run_benchmarks():
    """Run benchmarks for various configurations and print results."""
    results = []
    devices = ["cpu", "cuda"]
    dtypes = [torch.float32, torch.float64, torch.float16, torch.int32]
    formats = ["xyxy", "xywh", "cxcywh"]

    for device in devices:
        for dtype in dtypes:
            for in_fmt in formats:
                for out_fmt in formats:
                    if in_fmt != out_fmt:  # Skip same format conversions
                        result = benchmark_box_conversion(
                            device, dtype, in_fmt, out_fmt
                        )
                        results.append(result)

    df = pd.DataFrame([r.__dict__ for r in results])
    df["speedup"] = df["time_tv"] / df["time_tfbo"]
    print(df)
    print("\nAverage Speedup:", df["speedup"].mean())


if __name__ == "__main__":
    run_benchmarks()
