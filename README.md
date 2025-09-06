# torch-fast-box-ops

Fused CPU and CUDA kernels for faster bounding box operations.

## Build Requirements

- CUDA Toolkit (CTK) 12.8 or above (required for cuda::ceil_div convenience)
- PyTorch 2.4 or above

## Installation

Most reliable method is to pip install via github so it's compiled for your cuda toolkit, arch and python version.

```sh
pip3 install git+https://github.com/5had3z/torch-fast-box-ops --no-build-isolation
```

__TODO__: Look at PyTorch's work on cuda arch wheels and stable python abi, maybe then upload to pypi

### Building profile.cpp

[profile.cpp](./torch_fast_box_ops/profile.cpp) is mainly used to ad-hoc run one kernel with example input to profile with nsight compute. Often to build, we need to set some args to help cmake find the right CTK and torch. The following is an example of what needs to be added to `.vscode/settings.json`, if invoking in CLI then use the same definition declarations.

```json
{
  "cmake.configureArgs": [
    "-DTORCH_PATH:STRING=/path/to/lib/python3.12/site-packages/torch",
    "-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8",
    "-DUSE_SYSTEM_NVTX:BOOL=ON",
  ]
}
```

## Usage

Replace your torchvision.ops.boxes with torch_fast_box_ops!

```python
# from torchvision.ops.boxes import box_area, box_convert
from torch_fast_box_ops import box_area, box_convert

xywh = torch.ones((16,123,4), device="cuda")
xyxy = box_convert(xywh, in_fmt="xywh", out_fmt="xyxy")
area = box_area(xyxy)
```

## Benchmarks

Build/install the library and run the scripts in the [benchmarks](./benchmark/) to check out the speed ups from 0.1-20x. Most often there is a performance degredation with CPU and `torch.float16` and `torch.bfloat16` as CPUs don't have native scalar units to deal with this datatype and its simulated instead. I might add an escape hatch to run conversion first and then the op, and back to `[b]float16`. Some example results are shown below.

```text
Box IoU Benchmark Results:
  device          dtype  time_tfbo   time_tv    speedup
0    cpu  torch.float32   0.004539  0.020058   4.419119
1    cpu  torch.float64   0.003850  0.020625   5.356762
2    cpu  torch.float16   0.092905  0.020988   0.225908
3    cpu    torch.int32   0.002175  0.018344   8.432558
4   cuda  torch.float32   0.000658  0.013809  20.984500
5   cuda  torch.float64   0.000642  0.013852  21.582317
6   cuda  torch.float16   0.000624  0.016550  26.511678
7   cuda    torch.int32   0.000644  0.013808  21.430888
Average speedup: 13.617966220142184
```

```text
Generalized Box IoU Loss Benchmark Results:
  device          dtype  time_tfbo   time_tv   speedup
0    cpu  torch.float32   0.013625  0.047079  3.455420
1    cpu  torch.float64   0.011132  0.045750  4.109801
2    cpu  torch.float16   0.019775  0.062975  3.184585
3   cuda  torch.float32   0.019737  0.158772  8.044402
4   cuda  torch.float64   0.021202  0.159888  7.541249
5   cuda  torch.float16   0.020708  0.161895  7.817856

Average speedup: 5.692218825274298
```

```text
Box Conversion Results
   device          dtype  in_fmt out_fmt  time_tfbo   time_tv   speedup
0     cpu  torch.float32    xyxy    xywh   0.016899  0.036651  2.168888
1     cpu  torch.float32    xyxy  cxcywh   0.015573  0.057248  3.676201
2     cpu  torch.float32    xywh    xyxy   0.015028  0.037284  2.480941
3     cpu  torch.float32    xywh  cxcywh   0.015072  0.097199  6.448831
...
14    cpu  torch.float16    xywh    xyxy   0.076336  0.059051  0.773566
15    cpu  torch.float16    xywh  cxcywh   0.072429  0.160765  2.219624
16    cpu  torch.float16  cxcywh    xyxy   0.101918  0.130837  1.283754
...
24   cuda  torch.float32    xyxy    xywh   0.019932  0.026012  1.305005
25   cuda  torch.float32    xyxy  cxcywh   0.018579  0.047210  2.541063
26   cuda  torch.float32    xywh    xyxy   0.018278  0.024821  1.358007
27   cuda  torch.float32    xywh  cxcywh   0.018408  0.071097  3.862375
...
44   cuda    torch.int32    xywh    xyxy   0.018899  0.025642  1.356796
45   cuda    torch.int32    xywh  cxcywh   0.018630  0.088554  4.753387
46   cuda    torch.int32  cxcywh    xyxy   0.018943  0.061258  3.233862
47   cuda    torch.int32  cxcywh    xywh   0.018436  0.084666  4.592421
Average Speedup: 3.2261832083545507
```

System Spec: AMD Ryzen 9 5950X + RTX 3090

## Supported Operations

Plenty of operations to do, but the most common ones imo are covered, unsupported exotic loss functions can still benefit from using `_loss_inter_union`.

### General Operations

With autograd

- [x] `box_convert`
- [x] `box_area`

Without autograd

- [x] `box_iou`
- [x] `generalized_box_iou`
- [ ] `complete_box_iou`
- [x] `distance_box_iou`

### Loss Operations

Obviously this has to have autograd

- [x] `_loss_inter_union`
- [x] `generalized_box_iou_loss`
- [ ] `completed_box_iou_loss`
- [x] `distance_box_iou_loss`
- [ ] `minimum_points_distance_loss`
- [ ] `normalized_wasserstein_distance_loss`
- [ ] `inner_iou_loss`
