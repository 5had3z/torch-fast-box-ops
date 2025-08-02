
#include <torch/extension.h>

TORCH_LIBRARY(box_ops, m)
{
    // General Box Operations
    m.def("box_convert(Tensor input, str in_fmt, str out_fmt) -> Tensor");
    m.def("box_convert_backward(Tensor grad, str in_fmt, str out_fmt) -> Tensor");
    m.def("box_area(Tensor boxes) -> Tensor");
    m.def("box_area_backward(Tensor grad, Tensor boxes) -> Tensor");

    // IoU Inference operations fn(N,M)->NM, no backward
    m.def("box_iou(Tensor boxes1, Tensor boxes2) -> Tensor");
    m.def("generalized_box_iou(Tensor boxes1, Tensor boxes2) -> Tensor");

    // IoU Loss operations fn(N,N)->N, has backward
    m.def("_loss_inter_union(Tensor boxes1, Tensor boxes2) -> (Tensor, Tensor)");
    m.def(
        "_loss_inter_union_backward(Tensor grad_inter, Tensor grad_union, Tensor boxes1, Tensor boxes2) -> (Tensor, "
        "Tensor)");
    m.def("generalized_box_iou_loss(Tensor boxes1, Tensor boxes2, float eps) -> Tensor");
    m.def("generalized_box_iou_loss_backward(Tensor grad, Tensor boxes1, Tensor boxes2) -> (Tensor, Tensor)");
}
