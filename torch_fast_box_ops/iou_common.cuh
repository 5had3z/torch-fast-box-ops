#pragma once

#include "boxes.cuh"
#include "kernel.cuh"

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

template<typename T> TFBO_HOST_DEVICE auto box_area_op(const XYXY<T> &box) -> T
{
    return (box.x2 - box.x1) * (box.y2 - box.y1);
}

template<typename T> auto TFBO_HOST_DEVICE box_area_grad(XYXY<T> box) -> XYXY<T>
{
    XYXY<T> grad_box;
    grad_box.x1 = box.y1 - box.y2;
    grad_box.y1 = box.x1 - box.x2;
    grad_box.x2 = box.y2 - box.y1;
    grad_box.y2 = box.x2 - box.x1;
    return grad_box;
}

template<typename T> TFBO_HOST_DEVICE auto box_intersection(const XYXY<T> &box1, const XYXY<T> &box2) -> XYXY<T>
{
    XYXY<T> inter_box;
    inter_box.x1 = std::max(box1.x1, box2.x1);
    inter_box.y1 = std::max(box1.y1, box2.y1);
    inter_box.x2 = std::min(box1.x2, box2.x2);
    inter_box.y2 = std::min(box1.y2, box2.y2);
    return inter_box;
}


template<typename T> TFBO_HOST_DEVICE auto box_intersection_area(const XYXY<T> &box1, const XYXY<T> &box2) -> T
{
    auto inter_box = box_intersection(box1, box2);
    T inter_area = std::max(inter_box.x2 - inter_box.x1, static_cast<T>(0))
                   * std::max(inter_box.y2 - inter_box.y1, static_cast<T>(0));
    return inter_area;
}

/**
 * @brief Compute the minimum enclosing box of two boxes.
 *
 * @tparam T
 * @param box1
 * @param box2
 * @return XYXY<T> minumum enclosing box that contains both box1 and box2.
 */
template<typename T> TFBO_HOST_DEVICE auto min_enclosing_box(const XYXY<T> &box1, const XYXY<T> &box2) -> XYXY<T>
{
    XYXY<T> enclosing_box;
    enclosing_box.x1 = std::min(box1.x1, box2.x1);
    enclosing_box.y1 = std::min(box1.y1, box2.y1);
    enclosing_box.x2 = std::max(box1.x2, box2.x2);
    enclosing_box.y2 = std::max(box1.y2, box2.y2);
    return enclosing_box;
}

template<typename T> struct CXCY
{
    T cx;
    T cy;

    explicit TFBO_HOST_DEVICE CXCY(XYXY<T> box) noexcept
        : cx{ static_cast<T>(0.5) * (box.x1 + box.x2) }, cy{ static_cast<T>(0.5) * (box.y1 + box.y2) }
    {}
};

template<typename T> TFBO_HOST_DEVICE auto dist_sq(T p1, T p2) -> T { return p1 * p1 + p2 * p2; };