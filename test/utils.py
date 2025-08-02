import torch


def make_random_boxes(
    fmt: str,
    num_boxes: int,
    dtype: torch.dtype,
    device: str,
    normalized: bool = False,
    num_batch: int = 1,
    seed: int = 0,
) -> torch.Tensor:
    """Generate random bounding boxes in the specified format."""
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if fmt == "xyxy":
        rand_xy = torch.rand(num_batch, num_boxes, 2)
        rand_wh = torch.rand(num_batch, num_boxes, 2)
        if not normalized:
            rand_xy *= 100
            rand_wh *= 50
        boxes = torch.cat([rand_xy, rand_xy + rand_wh], dim=-1)
    else:
        boxes = torch.rand(num_batch, num_boxes, 4)
        if not normalized:
            boxes *= 100

    boxes = boxes.to(dtype=dtype, device=device)
    if num_batch == 1:
        boxes = boxes.squeeze(0)

    return boxes


def make_random_box_pairs(
    *args, seed_0: int = 0, seed_1: int = 1, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate pairs of random bounding boxes for testing.
    This is a convenience function that calls `make_random_boxes` twice.
    """
    boxes1 = make_random_boxes(*args, **kwargs, seed=seed_0)
    boxes2 = make_random_boxes(*args, **kwargs, seed=seed_1)
    return boxes1, boxes2
