#include <torch/torch.h>

#include <cxxopts.hpp>

auto box_iou(const torch::Tensor &boxes1, const torch::Tensor &boxes2) -> torch::Tensor;

int main(int argc, char *argv[])
{
    cxxopts::Options options("Profile Box Ops", "Run profiling for box operations in torch_fast_box_ops");

    auto boxes1 = torch::rand({ 16, 1000, 4 }, torch::kFloat32).cuda();
    auto boxes2 = torch::rand({ 16, 1000, 4 }, torch::kFloat32).cuda();

    auto iou = box_iou(boxes1, boxes2);
}
