import torch
def yolov8_collate_fn(batch):
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