
#include <torch/extension.h>

TORCH_LIBRARY(box_ops, m)
{
    m.def("box_iou(Tensor boxes1, Tensor boxes2) -> Tensor");
    m.def("_loss_inter_union(Tensor boxes1, Tensor boxes2) -> (Tensor, Tensor)");
    m.def(
        "_loss_inter_union_backward(Tensor grad_inter, Tensor grad_union, Tensor boxes1, Tensor boxes2) -> (Tensor, "
        "Tensor)");
    m.def("box_convert(Tensor input, str in_fmt, str out_fmt) -> Tensor");
    m.def("box_convert_backward(Tensor grad, str in_fmt, str out_fmt) -> Tensor");
    m.def("box_area(Tensor boxes) -> Tensor");
    m.def("box_area_backward(Tensor grad, Tensor boxes) -> Tensor");
}
