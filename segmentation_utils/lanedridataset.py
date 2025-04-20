import torch
from torch.utils.data import Dataset
import cv2
import os


class SegmentationDataset(Dataset):
    def __init__(self, image_dir,lane_mask_dir,drivable_area_dir,image_names, transform=None):
        """
        Args:
            image_dir (str): Directory with images.
            transform: An Albumentations Compose transform pipeline.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = image_names
        self.lane_mask_dir = lane_mask_dir
        self.drivable_area_dir = drivable_area_dir
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        image = self._load_image(image_name)
        lane_mask = self._load_mask(image_name, self.lane_mask_dir, "_lane_markings.png")
        drivable_mask = self._load_mask(image_name, self.drivable_area_dir, "_drivable_color.png")

        
        
        if self.transform:
            augmented = self.transform(image=image, masks = [lane_mask, drivable_mask])
            image = augmented["image"]
            lane_mask,drivable_mask = augmented["masks"]
        
       
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        lane_mask_tensor = torch.from_numpy(lane_mask).long()
        drivable_mask_tensor = torch.from_numpy(drivable_mask).long()

        
        return {
            "image": image_tensor,
            "lane_mask": lane_mask_tensor,
            "drivable_mask": drivable_mask_tensor
        }

    
    def _load_image(self, image_name):
        path = os.path.join(self.image_dir, f"{image_name}.jpg")
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_mask(self, image_name, base_dir, suffix):
        path = os.path.join(base_dir, f"{image_name}{suffix}")
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {path}")
        return mask