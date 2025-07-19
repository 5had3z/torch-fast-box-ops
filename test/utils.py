import torch


def make_random_boxes(
    fmt: str, num_boxes: int, dtype: torch.dtype, device: str, normalized: bool = False
) -> torch.Tensor:
    """Generate random bounding boxes in the specified format."""
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)

    if fmt == "xyxy":
        rand_xy = torch.rand(num_boxes, 2)
        rand_wh = torch.rand(num_boxes, 2)
        if not normalized:
            rand_xy *= 100
            rand_wh *= 50
        boxes = torch.cat([rand_xy, rand_xy + rand_wh], dim=-1)
    else:
        boxes = torch.rand(num_boxes, 4)
        if not normalized:
            boxes *= 100

    boxes = boxes.to(dtype=dtype, device=device)

    return boxes
