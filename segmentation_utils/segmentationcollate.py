import torch

def segmentation_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    lane_masks = torch.stack([item["lane_mask"] for item in batch])
    drivable_masks = torch.stack([item["drivable_mask"] for item in batch])

    
    return {
        "image": images,
        "lane_mask": lane_masks,
        "drivable_mask": drivable_masks
    }
