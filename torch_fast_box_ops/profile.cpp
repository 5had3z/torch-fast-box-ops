#include <cxxopts.hpp>
#include <torch/torch.h>

struct iou_type_tag
{
};
struct iou_tag : iou_type_tag
{
};
struct giou_tag : iou_type_tag
{
};
struct diou_tag : iou_type_tag
{
};
struct ciou_tag : iou_type_tag
{
};

template<typename IouType> auto box_iou(const torch::Tensor &boxes1, const torch::Tensor &boxes2) -> torch::Tensor;

auto box_convert_forward(const torch::Tensor &input, const std::string &in_fmt, const std::string &out_fmt)
    -> torch::Tensor;

int main(int argc, char *argv[])
{
    cxxopts::Options options("Profile Box Ops", "Run profiling for box operations in torch_fast_box_ops");

    auto boxes1 = torch::rand({ 16, 1000, 4 }, torch::kFloat32).cuda();
    auto boxes2 = torch::rand({ 16, 1000, 4 }, torch::kFloat32).cuda();

    auto iou = box_iou<iou_tag>(boxes1, boxes2);
    // box_convert_forward(boxes1, "cxcywh", "xyxy");
}
