#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda/cmath>

#include "iou_common.cuh"

auto loss_inter_union(const torch::Tensor &boxes1, const torch::Tensor &boxes2)
    -> std::tuple<torch::Tensor, torch::Tensor>
{
    TORCH_CHECK(boxes1.sizes() == boxes2.sizes(), "Input tensors boxes1 and boxes2 must have the same shape");
    TORCH_CHECK(boxes1.ndimension() == 2 && boxes1.size(-1) == 4, "Input tensors must have shape (N, 4)");
    const auto common_dtype = c10::promoteTypes(boxes1.scalar_type(), boxes2.scalar_type());
    auto boxes1_c = boxes1.contiguous().to(common_dtype);
    auto boxes2_c = boxes2.contiguous().to(common_dtype);

    torch::Tensor intersection = boxes1_c.new_empty({ boxes1.size(0) });
    torch::Tensor union_area = boxes1_c.new_empty({ boxes1.size(0) });

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(boxes1.scalar_type(), "_loss_inter_union", [&] {
        const auto num_boxes = boxes1.size(0);
        const auto boxes1_ptr = static_cast<const XYXY<scalar_t> *>(boxes1_c.const_data_ptr());
        const auto boxes2_ptr = static_cast<const XYXY<scalar_t> *>(boxes2_c.const_data_ptr());
        auto intersection_ptr = static_cast<scalar_t *>(intersection.mutable_data_ptr());
        auto union_area_ptr = static_cast<scalar_t *>(union_area.mutable_data_ptr());

        if (boxes1.is_cuda()) {
            auto kernel = [=] __device__(unsigned int idx) {
                intersection_ptr[idx] = box_intersection_area(boxes1_ptr[idx], boxes2_ptr[idx]);
                union_area_ptr[idx] =
                    box_area_op(boxes1_ptr[idx]) + box_area_op(boxes2_ptr[idx]) - intersection_ptr[idx];
            };
            launch_elementwise_kernel(kernel, num_boxes, at::cuda::getCurrentCUDAStream());
        } else {
            for (std::size_t i = 0; i < num_boxes; ++i) {
                intersection_ptr[i] = box_intersection_area(boxes1_ptr[i], boxes2_ptr[i]);
                union_area_ptr[i] = box_area_op(boxes1_ptr[i]) + box_area_op(boxes2_ptr[i]) - intersection_ptr[i];
            }
        }
    });

    return { intersection, union_area };
}

template<typename T>
TFBO_HOST_DEVICE auto intersection_grad(const XYXY<T> &box1, const XYXY<T> &box2, const XYXY<T> &inter_box)
    -> std::tuple<XYXY<T>, XYXY<T>>
{
    XYXY<T> grad_box1, grad_box2;

    T inter_width = inter_box.x2 - inter_box.x1;
    T inter_height = inter_box.y2 - inter_box.y1;
    bool okay = inter_width > 0 && inter_height > 0;
    inter_width *= okay ? 1 : 0;
    inter_height *= okay ? 1 : 0;
    const auto subgrad = static_cast<T>(0.5);

    bool x1_gt = box1.x1 > box2.x1;
    bool x1_eq = box1.x1 == box2.x1;
    bool x1_lt = box1.x1 < box2.x1;
    grad_box1.x1 = -(x1_gt + subgrad * x1_eq) * inter_height;
    grad_box2.x1 = -(x1_lt + subgrad * x1_eq) * inter_height;

    bool y1_gt = box1.y1 > box2.y1;
    bool y1_eq = box1.y1 == box2.y1;
    bool y1_lt = box1.y1 < box2.y1;
    grad_box1.y1 = -(y1_gt + subgrad * y1_eq) * inter_width;
    grad_box2.y1 = -(y1_lt + subgrad * y1_eq) * inter_width;

    bool x2_gt = box1.x2 > box2.x2;
    bool x2_eq = box1.x2 == box2.x2;
    bool x2_lt = box1.x2 < box2.x2;
    grad_box1.x2 = (x2_lt + subgrad * x2_eq) * inter_height;
    grad_box2.x2 = (x2_gt + subgrad * x2_eq) * inter_height;

    bool y2_gt = box1.y2 > box2.y2;
    bool y2_eq = box1.y2 == box2.y2;
    bool y2_lt = box1.y2 < box2.y2;
    grad_box1.y2 = (y2_lt + subgrad * y2_eq) * inter_width;
    grad_box2.y2 = (y2_gt + subgrad * y2_eq) * inter_width;

    return { grad_box1, grad_box2 };
}

template<typename T>
TFBO_HOST_DEVICE auto inter_union_grad(T grad_inter, T grad_union, const XYXY<T> &box1, const XYXY<T> &box2)
    -> std::tuple<XYXY<T>, XYXY<T>>
{
    XYXY<T> inter_box = box_intersection(box1, box2);
    T inter_area = std::max(box_area_op(inter_box), static_cast<T>(0));
    T union_area = box_area_op(box1) + box_area_op(box2) - inter_area;

    auto [inter_grad_box1, inter_grad_box2] = intersection_grad(box1, box2, inter_box);
    auto area_grad_box1 = box_area_grad(box1);
    auto area_grad_box2 = box_area_grad(box2);

    // dUnion = dArea1 + dArea2 - dIntersection
    // grad = dUnion * gradUnion + dIntersection * gradInter
    // grad = (dArea - dIntersection) * gradUnion + dIntersection * gradInter
    // grad = dArea * gradUnion + (gradInter - gradUnion) * dIntersection
    T grad_inter_ = grad_inter - grad_union;

    XYXY<T> grad_box1;
    grad_box1.x1 = grad_inter_ * inter_grad_box1.x1 + grad_union * area_grad_box1.x1;
    grad_box1.y1 = grad_inter_ * inter_grad_box1.y1 + grad_union * area_grad_box1.y1;
    grad_box1.x2 = grad_inter_ * inter_grad_box1.x2 + grad_union * area_grad_box1.x2;
    grad_box1.y2 = grad_inter_ * inter_grad_box1.y2 + grad_union * area_grad_box1.y2;

    XYXY<T> grad_box2;
    grad_box2.x1 = grad_inter_ * inter_grad_box2.x1 + grad_union * area_grad_box2.x1;
    grad_box2.y1 = grad_inter_ * inter_grad_box2.y1 + grad_union * area_grad_box2.y1;
    grad_box2.x2 = grad_inter_ * inter_grad_box2.x2 + grad_union * area_grad_box2.x2;
    grad_box2.y2 = grad_inter_ * inter_grad_box2.y2 + grad_union * area_grad_box2.y2;

    return { grad_box1, grad_box2 };
}

auto loss_inter_union_backward(const torch::Tensor &grad_inter,
    const torch::Tensor &grad_union,
    const torch::Tensor &boxes1,
    const torch::Tensor &boxes2) -> std::tuple<torch::Tensor, torch::Tensor>
{
    TORCH_CHECK(boxes1.sizes() == boxes2.sizes(), "Input tensors boxes1 and boxes2 must have the same shape");
    TORCH_CHECK(boxes1.ndimension() == 2 && boxes1.size(-1) == 4, "Input tensors must have shape (N, 4)");
    const auto common_dtype = c10::promoteTypes(c10::promoteTypes(grad_inter.scalar_type(), grad_union.scalar_type()),
        c10::promoteTypes(boxes1.scalar_type(), boxes2.scalar_type()));
    auto grad_inter_c = grad_inter.contiguous().to(common_dtype);
    auto grad_union_c = grad_union.contiguous().to(common_dtype);
    auto boxes1_c = boxes1.contiguous().to(common_dtype);
    auto boxes2_c = boxes2.contiguous().to(common_dtype);

    auto grad_boxes1 = torch::empty_like(boxes1_c);
    auto grad_boxes2 = torch::empty_like(boxes2_c);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(common_dtype, "_loss_inter_union_backward", [&] {
        const auto num_boxes = boxes1.size(0);
        const auto boxes1_ptr = static_cast<const XYXY<scalar_t> *>(boxes1_c.const_data_ptr());
        const auto boxes2_ptr = static_cast<const XYXY<scalar_t> *>(boxes2_c.const_data_ptr());
        auto grad_boxes1_ptr = static_cast<XYXY<scalar_t> *>(grad_boxes1.mutable_data_ptr());
        auto grad_boxes2_ptr = static_cast<XYXY<scalar_t> *>(grad_boxes2.mutable_data_ptr());
        const auto grad_inter_ptr = grad_inter_c.const_data_ptr<scalar_t>();
        const auto grad_union_ptr = grad_union_c.const_data_ptr<scalar_t>();

        if (boxes1.is_cuda()) {
            auto kernel = [=] __device__(unsigned int idx) {
                auto [grad_boxes1, grad_boxes2] =
                    inter_union_grad(grad_inter_ptr[idx], grad_union_ptr[idx], boxes1_ptr[idx], boxes2_ptr[idx]);
                grad_boxes1_ptr[idx] = grad_boxes1;
                grad_boxes2_ptr[idx] = grad_boxes2;
            };
            launch_elementwise_kernel(kernel, num_boxes, at::cuda::getCurrentCUDAStream());
        } else {
            for (std::size_t i = 0; i < num_boxes; ++i) {
                std::tie(grad_boxes1_ptr[i], grad_boxes2_ptr[i]) =
                    inter_union_grad(grad_inter_ptr[i], grad_union_ptr[i], boxes1_ptr[i], boxes2_ptr[i]);
            }
        }
    });

    return { grad_boxes1, grad_boxes2 };
}

template<typename T> TFBO_HOST_DEVICE auto iou_loss_fn(const XYXY<T> &box1, const XYXY<T> &box2, T eps, giou_tag) -> T
{
    auto intersection = box_intersection_area(box1, box2);
    auto union_area = box_area_op(box1) + box_area_op(box2) - intersection;
    XYXY<T> enclosing_box = min_enclosing_box(box1, box2);
    T enclosing_area = std::max(box_area_op(enclosing_box), static_cast<T>(0));
    T giou = intersection / union_area - (enclosing_area - union_area) / (enclosing_area + eps);
    return 1 - giou;
}

template<typename T>
TFBO_HOST_DEVICE auto iou_grad(T grad_loss, const XYXY<T> &box1, const XYXY<T> &box2, T eps, giou_tag)
    -> std::tuple<XYXY<T>, XYXY<T>>
{
    T inter_area = box_intersection_area(box1, box2);
    T union_area = box_area_op(box1) + box_area_op(box2) - inter_area;
    XYXY<T> enclosing_box = min_enclosing_box(box1, box2);
    T enc_area = std::max(box_area_op(enclosing_box), static_cast<T>(0));

    T enc_area_eps = enc_area + eps;
    T union_area_eps = union_area + eps;

    T grad_enc_area = grad_loss * union_area / (enc_area_eps * enc_area_eps);
    T grad_inter = -grad_loss / union_area_eps;
    T grad_union = grad_loss * (inter_area / (union_area_eps * union_area_eps) - 1 / enc_area_eps);

    auto [grad_box1_enc, grad_box2_enc] = min_enclosing_box_grad(box1, box2, enclosing_box);
    auto [grad_box1, grad_box2] = inter_union_grad(grad_inter, grad_union, box1, box2);

    // Combine gradients with FMA
    grad_box1.x1 = fma(grad_box1_enc.x1, grad_enc_area, grad_box1.x1);
    grad_box1.y1 = fma(grad_box1_enc.y1, grad_enc_area, grad_box1.y1);
    grad_box1.x2 = fma(grad_box1_enc.x2, grad_enc_area, grad_box1.x2);
    grad_box1.y2 = fma(grad_box1_enc.y2, grad_enc_area, grad_box1.y2);

    grad_box2.x1 = fma(grad_box2_enc.x1, grad_enc_area, grad_box2.x1);
    grad_box2.y1 = fma(grad_box2_enc.y1, grad_enc_area, grad_box2.y1);
    grad_box2.x2 = fma(grad_box2_enc.x2, grad_enc_area, grad_box2.x2);
    grad_box2.y2 = fma(grad_box2_enc.y2, grad_enc_area, grad_box2.y2);

    return { grad_box1, grad_box2 };
}

template<typename T> TFBO_HOST_DEVICE auto iou_loss_fn(const XYXY<T> &box1, const XYXY<T> &box2, T eps, diou_tag) -> T
{
    auto intersection = box_intersection_area(box1, box2);
    auto union_area = box_area_op(box1) + box_area_op(box2) - intersection;
    XYXY<T> enclosing_box = min_enclosing_box(box1, box2);
    const T diag_dist_sq = dist_sq<T>(enclosing_box.x2 - enclosing_box.x1, enclosing_box.y2 - enclosing_box.y1);
    const CXCY<T> box1c(box1);
    const CXCY<T> box2c(box2);
    const T cent_dist_sq = dist_sq<T>(box1c.cx - box2c.cx, box1c.cy - box2c.cy);
    return 1 - intersection / union_area + cent_dist_sq / (diag_dist_sq + static_cast<T>(1e-7));
}

/**
 * @brief  Gradient of a box corner p with respect to center distance squared value C -> dC/dp
 *         is a function of the box corner (x1), its opposite corner (x2) and the other box's center point (cx)
 *
 * @example gradient of box1 x1 is cdist_grad(box1.x1, box1.x2, box2.cx)
 *
 * @tparam T type of box points
 */
template<typename T> TFBO_HOST_DEVICE auto cdist_grad(T p1, T p2, T p3) -> T { return 0.5 * (p1 + p2) - p3; }


/**
 * @brief  Gradient of a top-left box corner p with respect to the enclosing box distance squared value
 *         D -> dC/dp is a function of the box corner (p1), the corner of the other box (p2) and
 *         the length of the enclosing box's side.
 *
 * @example gradient of box1 x1 is ddist_grad(box1.x1, box2.x1, enclosing_box_width)
 *
 * @tparam T type of box points
 * @param p1 point we want to find gradient of
 * @param p2 the other boxes same corner point
 * @param l smallest enclosing box length of the coordinate's dimension
 * @return the gradient of the box point p1
 */
template<typename T> TFBO_HOST_DEVICE auto ddist_grad_tl(T p1, T p2, T l) -> T
{
    T scale = 1 + static_cast<T>(p1 < p2);
    scale *= static_cast<T>(p1 <= p2);
    return -scale * l;
}

/**
 * @brief  Gradient of a bottom-right box corner p with respect to the enclosing box distance squared value
 *         D -> dC/dp is a function of the box corner (p1), the corner of the other box (p2) and
 *         the length of the enclosing box's side.
 *
 * @example gradient of box1 x2 is ddist_grad(box1.x2, box2.x2, enclosing_box_height)
 *
 * @tparam T type of box points
 * @param p1 point we want to find gradient of
 * @param p2 the other boxes same corner point
 * @param l the smallest enclosing box length of the coordinate's dimension
 * @return the gradient of the box point p1
 */
template<typename T> TFBO_HOST_DEVICE auto ddist_grad_br(T p1, T p2, T l) -> T
{
    T scale = 1 + static_cast<T>(p1 > p2);
    scale *= static_cast<T>(p1 >= p2);
    return scale * l;
}


template<typename T>
TFBO_HOST_DEVICE auto iou_grad(T grad_loss, const XYXY<T> &box1, const XYXY<T> &box2, T eps, diou_tag)
    -> std::tuple<XYXY<T>, XYXY<T>>
{
    const T inter_area = box_intersection_area(box1, box2);
    const T union_area = box_area_op(box1) + box_area_op(box2) - inter_area;
    const XYXY enclosing_box = min_enclosing_box(box1, box2);
    const CXCY<T> box1c(box1);
    const CXCY<T> box2c(box2);
    const T diag_dist_sq = dist_sq<T>(enclosing_box.x2 - enclosing_box.x1, enclosing_box.y2 - enclosing_box.y1);
    const T cent_dist_sq = dist_sq<T>(box1c.cx - box2c.cx, box1c.cy - box2c.cy);

    const T union_area_eps = union_area + eps;
    const T grad_inter = -grad_loss / union_area_eps;
    const T grad_union = grad_loss * inter_area / (union_area_eps * union_area_eps);

    const T grad_cent_dist = grad_loss / diag_dist_sq;
    const T grad_diag_dist = -grad_loss * cent_dist_sq / (diag_dist_sq * diag_dist_sq);

    auto [grad_box1, grad_box2] = inter_union_grad(grad_inter, grad_union, box1, box2);

    const T enc_w = enclosing_box.x2 - enclosing_box.x1;
    const T enc_h = enclosing_box.y2 - enclosing_box.y1;

    // Box 1 gradient
    // Center term
    grad_box1.x1 = fma(grad_cent_dist, cdist_grad(box1.x1, box1.x2, box2c.cx), grad_box1.x1);
    grad_box1.y1 = fma(grad_cent_dist, cdist_grad(box1.y1, box1.y2, box2c.cy), grad_box1.y1);
    grad_box1.x2 = fma(grad_cent_dist, cdist_grad(box1.x2, box1.x1, box2c.cx), grad_box1.x2);
    grad_box1.y2 = fma(grad_cent_dist, cdist_grad(box1.y2, box1.y1, box2c.cy), grad_box1.y2);
    // Diag term
    grad_box1.x1 = fma(grad_diag_dist, ddist_grad_tl(box1.x1, box2.x1, enc_w), grad_box1.x1);
    grad_box1.y1 = fma(grad_diag_dist, ddist_grad_tl(box1.y1, box2.y1, enc_h), grad_box1.y1);
    grad_box1.x2 = fma(grad_diag_dist, ddist_grad_br(box1.x2, box2.x2, enc_w), grad_box1.x2);
    grad_box1.y2 = fma(grad_diag_dist, ddist_grad_br(box1.y2, box2.y2, enc_h), grad_box1.y2);

    // Box 2 gradient
    // Center term
    grad_box2.x1 = fma(grad_cent_dist, cdist_grad(box2.x1, box2.x2, box1c.cx), grad_box2.x1);
    grad_box2.y1 = fma(grad_cent_dist, cdist_grad(box2.y1, box2.y2, box1c.cy), grad_box2.y1);
    grad_box2.x2 = fma(grad_cent_dist, cdist_grad(box2.x2, box2.x1, box1c.cx), grad_box2.x2);
    grad_box2.y2 = fma(grad_cent_dist, cdist_grad(box2.y2, box2.y1, box1c.cy), grad_box2.y2);

    // Diag term
    grad_box2.x1 = fma(grad_diag_dist, ddist_grad_tl(box2.x1, box1.x1, enc_w), grad_box2.x1);
    grad_box2.y1 = fma(grad_diag_dist, ddist_grad_tl(box2.y1, box1.y1, enc_h), grad_box2.y1);
    grad_box2.x2 = fma(grad_diag_dist, ddist_grad_br(box2.x2, box1.x2, enc_w), grad_box2.x2);
    grad_box2.y2 = fma(grad_diag_dist, ddist_grad_br(box2.y2, box1.y2, enc_h), grad_box2.y2);

    return { grad_box1, grad_box2 };
}


template<typename T> TFBO_HOST_DEVICE auto iou_loss_fn(const XYXY<T> &box1, const XYXY<T> &box2, T eps, ciou_tag) -> T
{
    const auto diou_loss = iou_loss_fn(box1, box2, eps, diou_tag{});
    const auto intersection = box_intersection_area(box1, box2);
    const auto union_area = box_area_op(box1) + box_area_op(box2) - intersection;
    const auto iou = intersection / union_area;

    const auto aspect =
        std::atan(box1.width() / (box1.height() + eps)) - std::atan(box2.width() / (box2.height() + eps));
    const auto v = 4 / (M_PI * M_PI) * aspect * aspect;
    const auto alpha = v / (1 - iou + v + eps);
    return diou_loss + alpha * v;
}

/**
 * @brief The dL/du factor of the box point gradients if we take v=4/(pi^2) * (u)^2
 *        where u=arctan(w1/h1) - arctan(w2/h2) is ratio_diff then dL/du = 4/(pi^2) * 2 * u
 *        and then we apply this to point p's `ciou_point_grad` (p=box1.x1 etc) dL/dp=dL/du * du/dp
 *
 * @tparam T datatype
 * @param alpha constant alpha term from CIoU
 * @param ratio_diff Aspect ratio difference of two boxes arctan(w1/h1) - arctan(w2/h2)
 * @return dL/du term
 */
template<typename T> TFBO_HOST_DEVICE auto ciou_ratio_grad(T alpha, T ratio_diff) -> T
{
    return 4 * alpha / (M_PI * M_PI) * 2 * ratio_diff;
}

/**
 * @brief Gradient calculation for CIoU aspect ratio difference term du/dp where u=arctan(w1/h1) - arctan(w2/h2)
 *        where p is a bounding box point (x1 or y2 etc).
 *
 * @note The signed-ness of this gradient depends on the box and corner. Where box1.x1 -ve, box1.y1 +ve and the
 *       inverse for bottom corner. Inverse this logic again again for box 2.
 *
 * @tparam T datatype
 * @param a The side length of the target dimension (width for x coords, height for ycoords)
 * @param b The other side's length
 * @return du/dp term
 */
template<typename T> TFBO_HOST_DEVICE auto ciou_point_grad(T a, T b) -> T { return b / (a * a + b * b); }


template<typename T>
TFBO_HOST_DEVICE auto iou_grad(T grad_loss, const XYXY<T> &box1, const XYXY<T> &box2, T eps, ciou_tag)
    -> std::tuple<XYXY<T>, XYXY<T>>
{
    const auto intersection = box_intersection_area(box1, box2);
    const auto union_area = box_area_op(box1) + box_area_op(box2) - intersection;
    const auto iou = intersection / union_area;

    const T aspect_diff =
        std::atan(box1.width() / (box1.height() + eps)) - std::atan(box2.width() / (box2.height() + eps));
    const T v = 4 / (M_PI * M_PI) * aspect_diff * aspect_diff;
    const T alpha = v / (1 - iou + v + eps);

    auto [box1grad, box2grad] = iou_grad(grad_loss, box1, box2, eps, diou_tag{});

    const auto ratio_grad = grad_loss * ciou_ratio_grad(alpha, aspect_diff);
    const auto box1_x_grad = ciou_point_grad(box1.width(), box1.height());
    const auto box1_y_grad = ciou_point_grad(box1.height(), box1.width());
    box1grad.x1 = fma(ratio_grad, -box1_x_grad, box1grad.x1);
    box1grad.y1 = fma(ratio_grad, box1_y_grad, box1grad.y1);
    box1grad.x2 = fma(ratio_grad, box1_x_grad, box1grad.x2);
    box1grad.y2 = fma(ratio_grad, -box1_y_grad, box1grad.y2);

    const auto box2_x_grad = ciou_point_grad(box2.width(), box2.height());
    const auto box2_y_grad = ciou_point_grad(box2.height(), box2.width());
    box2grad.x1 = fma(ratio_grad, box2_x_grad, box2grad.x1);
    box2grad.y1 = fma(ratio_grad, -box2_y_grad, box2grad.y1);
    box2grad.x2 = fma(ratio_grad, -box2_x_grad, box2grad.x2);
    box2grad.y2 = fma(ratio_grad, box2_y_grad, box2grad.y2);

    return { box1grad, box2grad };
}

template<typename IoUType>
auto box_iou_loss(const torch::Tensor &boxes1, const torch::Tensor &boxes2, double eps) -> torch::Tensor
{
    TORCH_CHECK(boxes1.sizes() == boxes2.sizes(), "Input tensors boxes1 and boxes2 must have the same shape");
    TORCH_CHECK(boxes1.ndimension() == 2 && boxes1.size(-1) == 4, "Input tensors must have shape (N, 4)");
    const auto common_dtype = c10::promoteTypes(boxes1.scalar_type(), boxes2.scalar_type());
    auto boxes1_c = boxes1.contiguous().to(common_dtype);
    auto boxes2_c = boxes2.contiguous().to(common_dtype);

    auto loss = boxes1_c.new_empty({ boxes1.size(0) });

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(common_dtype, "box_iou_loss", [&] {
        const auto num_boxes = boxes1.size(0);
        const auto boxes1_ptr = static_cast<const XYXY<scalar_t> *>(boxes1_c.const_data_ptr());
        const auto boxes2_ptr = static_cast<const XYXY<scalar_t> *>(boxes2_c.const_data_ptr());
        auto loss_ptr = loss.mutable_data_ptr<scalar_t>();
        const auto eps_t = static_cast<scalar_t>(eps);

        if (boxes1.is_cuda()) {
            auto kernel = [=] __device__(unsigned int idx) {
                loss_ptr[idx] = iou_loss_fn(boxes1_ptr[idx], boxes2_ptr[idx], eps_t, IoUType{});
            };
            launch_elementwise_kernel(kernel, num_boxes, at::cuda::getCurrentCUDAStream());
        } else {
            for (std::size_t i = 0; i < num_boxes; ++i) {
                loss_ptr[i] = iou_loss_fn(boxes1_ptr[i], boxes2_ptr[i], eps_t, IoUType{});
            }
        }
    });

    return loss;
}

template<typename IoUType>
auto box_iou_loss_backward(const torch::Tensor &grad,
    const torch::Tensor &boxes1,
    const torch::Tensor &boxes2,
    double eps) -> std::tuple<torch::Tensor, torch::Tensor>
{
    TORCH_CHECK(boxes1.sizes() == boxes2.sizes(), "Input tensors boxes1 and boxes2 must have the same shape");
    TORCH_CHECK(boxes1.ndimension() == 2 && boxes1.size(-1) == 4, "Input tensors must have shape (N, 4)");
    auto common_dtype =
        c10::promoteTypes(grad.scalar_type(), c10::promoteTypes(boxes1.scalar_type(), boxes2.scalar_type()));
    auto grad_c = grad.contiguous().to(common_dtype);
    auto boxes1_c = boxes1.contiguous().to(common_dtype);
    auto boxes2_c = boxes2.contiguous().to(common_dtype);

    auto grad_boxes1 = torch::empty_like(boxes1_c);
    auto grad_boxes2 = torch::empty_like(boxes2_c);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(common_dtype, "box_iou_loss_backward", [&] {
        const auto num_boxes = boxes1.size(0);
        const auto boxes1_ptr = static_cast<const XYXY<scalar_t> *>(boxes1_c.const_data_ptr());
        const auto boxes2_ptr = static_cast<const XYXY<scalar_t> *>(boxes2_c.const_data_ptr());
        auto grad_boxes1_ptr = static_cast<XYXY<scalar_t> *>(grad_boxes1.mutable_data_ptr());
        auto grad_boxes2_ptr = static_cast<XYXY<scalar_t> *>(grad_boxes2.mutable_data_ptr());
        const auto grad_ptr = grad_c.const_data_ptr<scalar_t>();
        auto eps_t = static_cast<scalar_t>(eps);
        if (boxes1.is_cuda()) {
            auto kernel = [=] __device__(unsigned int idx) {
                auto [grad_box1, grad_box2] =
                    iou_grad(grad_ptr[idx], boxes1_ptr[idx], boxes2_ptr[idx], eps_t, IoUType{});
                grad_boxes1_ptr[idx] = grad_box1;
                grad_boxes2_ptr[idx] = grad_box2;
            };
            launch_elementwise_kernel(kernel, num_boxes, at::cuda::getCurrentCUDAStream());
        } else {
            for (std::size_t i = 0; i < num_boxes; ++i) {
                std::tie(grad_boxes1_ptr[i], grad_boxes2_ptr[i]) =
                    iou_grad(grad_ptr[i], boxes1_ptr[i], boxes2_ptr[i], eps_t, IoUType{});
            }
        }
    });

    return { grad_boxes1, grad_boxes2 };
}

template<typename T>
TFBO_HOST_DEVICE auto min_enclosing_box_grad(const XYXY<T> &box1, const XYXY<T> &box2, const XYXY<T> &enc_box)
    -> std::tuple<XYXY<T>, XYXY<T>>
{
    XYXY<T> box1_grad, box2_grad;
    T enc_w = enc_box.x2 - enc_box.x1;
    T enc_h = enc_box.y2 - enc_box.y1;
    const T subgrad = static_cast<T>(0.5);

    bool x1_lt = box1.x1 < box2.x1;
    bool x1_eq = box1.x1 == box2.x1;
    bool x1_gt = box1.x1 > box2.x1;
    box1_grad.x1 = -(x1_lt + subgrad * x1_eq) * enc_h;
    box2_grad.x1 = -(x1_gt + subgrad * x1_eq) * enc_h;

    bool y1_lt = box1.y1 < box2.y1;
    bool y1_eq = box1.y1 == box2.y1;
    bool y1_gt = box1.y1 > box2.y1;
    box1_grad.y1 = -(y1_lt + subgrad * y1_eq) * enc_w;
    box2_grad.y1 = -(y1_gt + subgrad * y1_eq) * enc_w;

    bool x2_lt = box1.x2 < box2.x2;
    bool x2_eq = box1.x2 == box2.x2;
    bool x2_gt = box1.x2 > box2.x2;
    box1_grad.x2 = (x2_gt + subgrad * x2_eq) * enc_h;
    box2_grad.x2 = (x2_lt + subgrad * x2_eq) * enc_h;

    bool y2_lt = box1.y2 < box2.y2;
    bool y2_eq = box1.y2 == box2.y2;
    bool y2_gt = box1.y2 > box2.y2;
    box1_grad.y2 = (y2_gt + subgrad * y2_eq) * enc_w;
    box2_grad.y2 = (y2_lt + subgrad * y2_eq) * enc_w;

    return { box1_grad, box2_grad };
}


TORCH_LIBRARY_IMPL(box_ops, CPU, m)
{
    m.impl("_loss_inter_union", &loss_inter_union);
    m.impl("_loss_inter_union_backward", &loss_inter_union_backward);

    m.impl("generalized_box_iou_loss", &box_iou_loss<giou_tag>);
    m.impl("generalized_box_iou_loss_backward", &box_iou_loss_backward<giou_tag>);

    m.impl("distance_box_iou_loss", &box_iou_loss<diou_tag>);
    m.impl("distance_box_iou_loss_backward", &box_iou_loss_backward<diou_tag>);

    m.impl("complete_box_iou_loss", &box_iou_loss<ciou_tag>);
    m.impl("complete_box_iou_loss_backward", &box_iou_loss_backward<ciou_tag>);
}

TORCH_LIBRARY_IMPL(box_ops, CUDA, m)
{
    m.impl("_loss_inter_union", &loss_inter_union);
    m.impl("_loss_inter_union_backward", &loss_inter_union_backward);

    m.impl("generalized_box_iou_loss", &box_iou_loss<giou_tag>);
    m.impl("generalized_box_iou_loss_backward", &box_iou_loss_backward<giou_tag>);

    m.impl("distance_box_iou_loss", &box_iou_loss<diou_tag>);
    m.impl("distance_box_iou_loss_backward", &box_iou_loss_backward<diou_tag>);

    m.impl("complete_box_iou_loss", &box_iou_loss<ciou_tag>);
    m.impl("complete_box_iou_loss_backward", &box_iou_loss_backward<ciou_tag>);
}
