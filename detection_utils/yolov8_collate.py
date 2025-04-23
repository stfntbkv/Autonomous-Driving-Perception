import torch
def yolov8_collate_fn(batch):
    """
    Custom collate function for YOLOv8-style object detection models.

    This function processes a batch of image samples and their corresponding annotations
    to prepare them for training and validation. It handles:
      - Stacking input images into a single batch tensor,
      - Converting bounding boxes from (xyxy) to (xywh) format (as required by YOLOv8),
      - Creating additional metadata for validation (absolute coordinate targets),
      - Constructing target tensors for model training and evaluation.

    Args:
        batch (List[Dict]): List of dictionaries, each with:
            - 'image': Tensor of shape (3, H, W),
            - 'bboxes': Tensor of shape (N, 4), in normalized (xyxy) format,
            - 'labels': Tensor of shape (N,), containing class indices.

    Returns:
        Dict[str, Union[Tensor, List[Dict]]]: A dictionary with:
            - 'img': Tensor of shape (B, 3, H, W), stacked images.
            - 'batch_idx': Tensor of shape (total_boxes,), batch index for each box.
            - 'cls': Tensor of shape (total_boxes,), class labels for each box.
            - 'bboxes': Tensor of shape (total_boxes, 4), boxes in (cx, cy, w, h) format.
            - 'targets_for_metric': List of length B with dicts: {'boxes': Tensor [N, 4], 'labels': Tensor [N]}.
            - 'img_shape': Tuple[int, int], height and width of the input images.

    Note:
        Bounding boxes are assumed to be normalized and in (xyxy) format initially.
        The function handles empty boxes gracefully and ensures compatibility with both
        training loss and validation metric computation.
    """
    
    # Stack images
    images = torch.stack([item["image"] for item in batch])
    img_height, img_width = images.shape[2], images.shape[3]

    # Process detection targets
    all_batch_idx = []
    all_cls = []
    all_bboxes = []

    for batch_idx, item in enumerate(batch):
        bboxes = item["bboxes"]  # [N, 4] (xmin, ymin, xmax, ymax)
        labels = item["labels"]  # [N]

        if bboxes.numel() == 0:
            continue

        # Convert to xywh
        boxes_xywh = torch.zeros_like(bboxes)
        boxes_xywh[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2  # center x
        boxes_xywh[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2  # center y
        boxes_xywh[:, 2] = bboxes[:, 2] - bboxes[:, 0]       # width
        boxes_xywh[:, 3] = bboxes[:, 3] - bboxes[:, 1]       # height

        
        # Collect data
        all_bboxes.append(boxes_xywh)
        all_batch_idx.append(torch.full((labels.size(0),), batch_idx, dtype=torch.float32))
        all_cls.append(labels.float())  # YOLOv8 expects cls as float, cast to long later if needed

    # Combine tensors
    if all_bboxes:
        combined_bboxes = torch.cat(all_bboxes, 0)        # [total_boxes, 4]
        combined_batch_idx = torch.cat(all_batch_idx, 0)  # [total_boxes]
        combined_cls = torch.cat(all_cls, 0)              # [total_boxes]
    else:
        combined_bboxes = torch.empty((0, 4), dtype=torch.float32)
        combined_batch_idx = torch.empty((0,), dtype=torch.float32)
        combined_cls = torch.empty((0,), dtype=torch.float32)

    validation_targets = []
    for item in batch:
        bboxes = item["bboxes"]  # [N, 4] normalized xyxy
        labels = item["labels"]  # [N]
        # Scale to absolute coordinates using broadcasting
        scales = torch.tensor([img_width, img_height, img_width, img_height], device=bboxes.device)
        bboxes_abs = bboxes * scales  # [N, 4] absolute xyxy
        validation_targets.append({
            "boxes": bboxes_abs,
            "labels": labels
        })
        
    return {
        "img": images,
        "batch_idx": combined_batch_idx,
        "cls": combined_cls,
        "bboxes": combined_bboxes,
        "targets_for_metric": validation_targets,
        "img_shape": (img_height, img_width)
    }