"""Benchmark for box IoU operations"""

import functools
import timeit
from dataclasses import dataclass

import pandas as pd
import torch
from torchvision.ops._utils import _loss_inter_union as tv_loss_inter_union
from torchvision.ops.boxes import box_area as tv_box_area
from torchvision.ops.boxes import box_iou as tv_box_iou
from torchvision.ops.diou_loss import distance_box_iou_loss as tv_distance_box_iou_loss
from torchvision.ops.giou_loss import (
    generalized_box_iou_loss as tv_generalized_box_iou_loss,
)

from torch_fast_box_ops import _loss_inter_union as tfbo_loss_inter_union
from torch_fast_box_ops import box_area as tfbo_box_area
from torch_fast_box_ops import box_iou as tfbo_box_iou
from torch_fast_box_ops import distance_box_iou_loss as tfbo_distance_box_iou_loss
from torch_fast_box_ops import generalized_box_iou_loss as tfbo_generalized_box_iou_loss


@dataclass
class BoxIoUResult:
    device: str
    dtype: torch.dtype
    time_tfbo: float
    time_tv: float


def benchmark_box_area(device: str, dtype: torch.dtype, num_boxes: int = 10000):
    """
    Benchmark the box area operation.

    Args:
        device (str): Device to run the benchmark on ('cpu' or 'cuda').
        dtype (torch.dtype): Data type of the bounding boxes.
        num_boxes (int): Number of random boxes to generate for benchmarking.
    """
    torch.manual_seed(0)
    boxes = torch.rand(num_boxes, 4) * 100
    boxes = boxes.to(dtype=dtype, device=device)

    # Warm-up
    _ = tfbo_box_area(boxes)
    _ = tv_box_area(boxes)

    # Benchmark
    tfbo = timeit.timeit(lambda: tfbo_box_area(boxes), number=1000)
    tv = timeit.timeit(lambda: tv_box_area(boxes), number=1000)

    return BoxIoUResult(device=device, dtype=dtype, time_tfbo=tfbo, time_tv=tv)


def benchmark_box_iou(device: str, dtype: torch.dtype, num_boxes: int = 100):
    """
    Benchmark the box IoU operation.

    Args:
        device (str): Device to run the benchmark on ('cpu' or 'cuda').
        dtype (torch.dtype): Data type of the bounding boxes.
        num_boxes (int): Number of random boxes to generate for benchmarking.
    """
    torch.manual_seed(0)
    boxes1 = torch.rand(num_boxes, 4) * 100
    boxes2 = torch.rand(num_boxes * 2, 4) * 100
    boxes1 = boxes1.to(dtype=dtype, device=device)
    boxes2 = boxes2.to(dtype=dtype, device=device)

    # Warm-up
    _ = tfbo_box_iou(boxes1, boxes2)
    _ = tv_box_iou(boxes1, boxes2)

    # Benchmark
    tfbo = timeit.timeit(lambda: tfbo_box_iou(boxes1, boxes2), number=100)
    tv = timeit.timeit(lambda: tv_box_iou(boxes1, boxes2), number=100)

    return BoxIoUResult(device=device, dtype=dtype, time_tfbo=tfbo, time_tv=tv)


def benchmark_box_loss_inter_union(
    device: str, dtype: torch.dtype, num_boxes: int = 100
):
    """
    Benchmark the loss intersection and union operation.

    Args:
        device (str): Device to run the benchmark on ('cpu' or 'cuda').
        dtype (torch.dtype): Data type of the bounding boxes.
        num_boxes (int): Number of random boxes to generate for benchmarking.
    """
    torch.manual_seed(0)
    boxes1 = torch.rand(num_boxes, 4) * 100
    boxes2 = torch.rand(num_boxes, 4) * 100
    boxes1 = boxes1.to(dtype=dtype, device=device)
    boxes1.requires_grad = True
    boxes2 = boxes2.to(dtype=dtype, device=device)
    boxes2.requires_grad = True

    # Warm-up
    _ = tfbo_loss_inter_union(boxes1, boxes2)
    _ = tv_loss_inter_union(boxes1, boxes2)

    def _test_loss_inter_union(fn):
        inter, union = fn(boxes1, boxes2)
        loss = (1 - inter / union).mean()
        loss.backward()
        if boxes1.device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark
    tfbo = timeit.timeit(
        lambda: _test_loss_inter_union(tfbo_loss_inter_union), number=100
    )
    tv = timeit.timeit(lambda: _test_loss_inter_union(tv_loss_inter_union), number=100)

    return BoxIoUResult(device=device, dtype=dtype, time_tfbo=tfbo, time_tv=tv)


def benchmark_box_iou_loss(
    device: str, dtype: torch.dtype, iou_type: str, num_boxes: int = 100
):
    """
    Benchmark the generalized box IoU loss operation.

    Args:
        device (str): Device to run the benchmark on ('cpu' or 'cuda').
        dtype (torch.dtype): Data type of the bounding boxes.
        num_boxes (int): Number of random boxes to generate for benchmarking.
    """
    torch.manual_seed(0)
    boxes1 = torch.rand(num_boxes, 4) * 100
    boxes2 = torch.rand(num_boxes, 4) * 100
    boxes1 = boxes1.to(dtype=dtype, device=device)
    boxes1.requires_grad = True
    boxes2 = boxes2.to(dtype=dtype, device=device)
    boxes2.requires_grad = True

    tfbo_fn = {
        "giou": tfbo_generalized_box_iou_loss,
        "diou": tfbo_distance_box_iou_loss,
    }[iou_type]
    tv_fn = {"giou": tv_generalized_box_iou_loss, "diou": tv_distance_box_iou_loss}[
        iou_type
    ]

    # Warm-up
    _ = tfbo_fn(boxes1, boxes2)
    _ = tv_fn(boxes1, boxes2)

    def _test_iou_loss(fn):
        loss: torch.Tensor = fn(boxes1, boxes2)
        loss.mean().backward()
        if boxes1.device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark
    tfbo = timeit.timeit(lambda: _test_iou_loss(tfbo_fn), number=100)
    tv = timeit.timeit(lambda: _test_iou_loss(tv_fn), number=100)

    return BoxIoUResult(device=device, dtype=dtype, time_tfbo=tfbo, time_tv=tv)


def run_benchmark(devices: list[str], dtypes: list[torch.dtype], name: str, func):
    """
    Run a benchmark for the specified function across multiple devices and dtypes.

    Args:
        devices (list[str]): List of devices to run the benchmark on.
        dtypes (list[torch.dtype]): List of data types to test.
        name (str): Name of the benchmark for logging.
        func (callable): Benchmark function to execute.
    """
    results = []
    for device in devices:
        for dtype in dtypes:
            result = func(device, dtype)
            results.append(result)

    df = pd.DataFrame([r.__dict__ for r in results])
    df["speedup"] = df["time_tv"] / df["time_tfbo"]
    print(f"{name} Benchmark Results:")
    print(df)
    print("\nAverage speedup:", df["speedup"].mean())


def run_benchmarks():
    """Run benchmarks for various configurations and print results."""
    devices = ["cpu", "cuda"]
    dtypes = [torch.float32, torch.float64, torch.float16, torch.int32]

    run_benchmark(devices, dtypes, "Box Area", benchmark_box_area)

    run_benchmark(devices, dtypes, "Box IoU", benchmark_box_iou)

    no_int32_dtypes = [d for d in dtypes if d != torch.int32]
    run_benchmark(
        devices,
        no_int32_dtypes,
        "Loss Intersection/Union",
        benchmark_box_loss_inter_union,
    )

    run_benchmark(
        devices,
        no_int32_dtypes,
        "Generalized Box IoU Loss",
        functools.partial(benchmark_box_iou_loss, iou_type="giou"),
    )

    run_benchmark(
        devices,
        no_int32_dtypes,
        "Distance Box IoU Loss",
        functools.partial(benchmark_box_iou_loss, iou_type="diou"),
    )


if __name__ == "__main__":
    run_benchmarks()
