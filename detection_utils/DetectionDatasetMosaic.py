import os
import cv2
import math
import random
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
import albumentations as A

class Instances:
    """
    Lightweight bounding box container for object detection pipelines.

    This class is a simplified version of Ultralytics' Instances and focuses exclusively on bounding boxes.
    It provides a convenient wrapper for manipulating boxes across formats and coordinate systems (e.g., 
    normalized vs. absolute), and supports transformations commonly needed during training and augmentation.

    Attributes:
        bboxes (np.ndarray): Array of shape (N, 4) containing bounding boxes.
        bbox_format (str): Format of the bounding boxes ('xyxy' or 'xywh').
        normalized (bool): Whether the bounding boxes are normalized to [0, 1].

    Supported Operations:
        - Format conversion: between 'xyxy' and 'xywh'.
        - Normalization and denormalization based on image size.
        - Padding and clipping bounding boxes to image bounds.
        - Indexing to retrieve subsets of boxes.
        - Area-based filtering and scaling.

    Example:
        >>> instances = Instances([[10, 20, 100, 200]], bbox_format='xyxy')
        >>> instances.normalize(w=640, h=640)
        >>> instances.denormalize(w=640, h=640)
    """

    def __init__(self, bboxes, bbox_format='xyxy', normalized=False):
        """Initialize Instances object with bounding boxes, format, and normalization status."""
        if isinstance(bboxes, list):
            bboxes = np.array(bboxes, dtype=np.float32)
        elif isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()


        if bboxes.ndim == 1 and bboxes.shape[0] == 0:
           bboxes = np.empty((0, 4), dtype=np.float32) 
        elif bboxes.ndim != 2 or bboxes.shape[1] != 4:
             raise ValueError(f"Input bboxes have shape {bboxes.shape}")
                  


        self.bboxes = bboxes
        self.bbox_format = bbox_format
        self.normalized = normalized

    def __len__(self):
        """Return the number of instances."""
        return len(self.bboxes)

    def __getitem__(self, index):
        """Retrieve instance(s) by index. Returns a *new* Instances object."""
        if isinstance(index, (int, np.integer)): 
             new_bboxes = self.bboxes[index:index+1]
        elif isinstance(index, (slice, list, np.ndarray)):
             new_bboxes = self.bboxes[index]
        else:
            raise TypeError(f"Instances index must be int, slice, list, or np.ndarray, not {type(index)}")
      
        return Instances(new_bboxes, bbox_format=self.bbox_format, normalized=self.normalized)


    def convert_bbox(self, format='xyxy'):
        """Convert bounding box format."""
        if format == self.bbox_format or len(self.bboxes) == 0: return
        
        if format == 'xywh' and self.bbox_format == 'xyxy':
            x1, y1, x2, y2 = self.bboxes.T; self.bboxes = np.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), axis=1)
        elif format == 'xyxy' and self.bbox_format == 'xywh':
            xc, yc, w, h = self.bboxes.T; self.bboxes = np.stack((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2), axis=1)
        else: print(f"Warning: Bbox conversion from {self.bbox_format} to {format} not implemented."); return
        self.bbox_format = format
       

    def denormalize(self, w, h):
         """Denormalize bounding boxes if they are normalized."""
         if self.normalized and len(self.bboxes) > 0:
             self.convert_bbox(format='xyxy'); self.bboxes[:, [0, 2]] *= w; self.bboxes[:, [1, 3]] *= h; self.normalized = False

    def normalize(self, w, h):
         """Normalize bounding boxes if they are not normalized."""
         if not self.normalized and len(self.bboxes) > 0:
             self.convert_bbox(format='xyxy')
             # Add check for valid w, h
             if w > 0 and h > 0:
                  self.bboxes[:, [0, 2]] /= w; self.bboxes[:, [1, 3]] /= h
                  self.bboxes = self.bboxes.clip(0.0, 1.0); self.normalized = True
             else:
                  print(f"Warning: Invalid dimensions W:{w} H:{h} for normalization. Skipping.")


    def add_padding(self, padw, padh):
        """Add padding offset to bounding boxes."""
        if not self.normalized and len(self.bboxes) > 0:
            self.convert_bbox(format='xyxy'); self.bboxes[:, [0, 2]] += padw; self.bboxes[:, [1, 3]] += padh

    def clip(self, w, h):
        """Clip bounding boxes to image dimensions."""
        if len(self.bboxes) > 0:
            self.convert_bbox(format='xyxy'); self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w); self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)

    def scale(self, scale_w, scale_h, bbox_only=True): 
        """Scale bounding boxes. Assumes scaling absolute coordinates."""
        if len(self.bboxes) > 0:
             was_normalized = self.normalized
             if was_normalized: self.denormalize(1, 1) 
             self.convert_bbox(format='xyxy'); self.bboxes[:, [0, 2]] *= scale_w; self.bboxes[:, [1, 3]] *= scale_h
             if was_normalized: self.normalize(1, 1) 

    def remove_zero_area_boxes(self, wh_thr=2):
        """Remove boxes with zero or small area."""
        if len(self.bboxes) == 0: return np.array([], dtype=bool)
        self.convert_bbox(format='xyxy')
        box_w = self.bboxes[:, 2] - self.bboxes[:, 0]; box_h = self.bboxes[:, 3] - self.bboxes[:, 1]
        keep = (box_w > wh_thr) & (box_h > wh_thr)
        return keep


class RandomPerspective:
    """
    Applies random perspective and affine transformations to an image and its bounding boxes.

    This class is a stripped-down and custom version of the augmentation used in Ultralytics' YOLO pipelines.
    It introduces controlled randomness in rotation, translation, scaling, shearing, and perspective, which
    helps improve the generalization of object detection models during training.

    This version is adapted to work with custom `Instances` objects and assumes all label input 
    is structured as a dictionary containing image, class labels, and bounding boxes.

    Args:
        degrees (float): Max degrees for random rotation.
        translate (float): Maximum translation as a fraction of image dimensions.
        scale (float): Scaling factor range for zoom in/out.
        shear (float): Maximum shearing factor in degrees.
        perspective (float): Perspective distortion coefficient.
        border (Tuple[int, int]): Optional mosaic border padding (y, x).

    Call Args (dict):
        - 'img': The input image (H x W x C).
        - 'cls': Array of class indices.
        - 'instances': An Instances object with bounding boxes.
        - 'mosaic_border' (optional): Border padding for mosaic images.

    Returns:
        dict: Transformed image, filtered class labels, updated bounding boxes, and resized shape.

    Example:
        >>> transformer = RandomPerspective(degrees=10, translate=0.1)
        >>> augmented = transformer({'img': img, 'cls': labels, 'instances': boxes})
    """
    def __init__(self,degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,border=(0, 0)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective

    def __call__(self, labels):
        """Applies random perspective transformation."""
        img = labels['img']
        cls = labels['cls']
        instances = labels.pop('instances')

       
        instances.convert_bbox(format='xyxy')
        instances.denormalize(1, 1)

        border = labels.pop('mosaic_border', (0, 0))
        h, w = img.shape[:2]
        self.size = (w + border[1] * 2, h + border[0] * 2)

   
        img_transformed, M, scale_factor = self.affine_transform(img, border)

   
        bboxes_original_scaled = instances.bboxes.copy()
        if self.scale != 0: 
             bboxes_original_scaled[:, [0, 2]] *= scale_factor
             bboxes_original_scaled[:, [1, 3]] *= scale_factor

        bboxes_transformed = self.apply_bboxes(instances.bboxes, M)




        new_instances = Instances(bboxes=bboxes_transformed,
                                  
                                  bbox_format="xyxy",
                                  normalized=False)

        new_instances.clip(*self.size)

    
        keep = self.box_candidates(box1=bboxes_original_scaled.T, box2=new_instances.bboxes.T, area_thr=0.10)
        final_instances = new_instances[keep] 
        final_cls = cls[keep]


        output_labels = {}
        output_labels['img'] = img_transformed
        output_labels['cls'] = final_cls
        output_labels['instances'] = final_instances 
        output_labels['resized_shape'] = img_transformed.shape[:2] 

        return output_labels

  
    def affine_transform(self, img, border):
        """Applies affine transformations."""
      
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)

        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)

        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0] # Use self.size[0] (width)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1] # Use self.size[1] (height)

        M = T @ S @ R @ P @ C
        # Use self.size for dsize
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
            else:
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))
        return img, M, s 

    def apply_bboxes(self, bboxes, M):
        """Transforms bounding boxes using the affine matrix M."""
        n = len(bboxes)
        if n == 0:
            return bboxes
        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
       
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        xy = xy @ M.T
       
        if self.perspective:
            
            xy[:, :2] /= (xy[:, 2:3] + 1e-9) 
        xy = xy[:, :2].reshape(n, 8)
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]

        new_bboxes = np.concatenate(
            (x.min(1, keepdims=True), y.min(1, keepdims=True), x.max(1, keepdims=True), y.max(1, keepdims=True)), axis=1
        ).astype(bboxes.dtype)
        return new_bboxes



    @staticmethod
    def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
        """Filter candidates based on size, aspect ratio, and area ratio."""
 
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr) 





class DetectionDatasetMosaic(Dataset):
    """
    A custom PyTorch Dataset class for object detection with support for YOLO-style mosaic augmentation
    and advanced preprocessing, including optional post-transforms, letterboxing, and validation-friendly paths.

    This dataset handles:
    - Standard single-image loading with letterbox resizing,
    - Mosaic augmentation: combining 4 images into a single training sample,
    - Random perspective transformation (affine & perspective),
    - Optional Albumentations post-transforms (e.g., normalization, color jitter),
    - Normalized bounding box output in [0,1] format for compatibility with YOLO-style models,
    - Special treatment for validation mode (no augmentations, no mosaic).

    Args:
        image_dir (str): Directory containing the image files (assumed to be .jpg).
        bboxes_dic (dict): Dictionary of annotations with image filenames as keys and values containing
                           "boxes" and "labels".
        degrees (float): Max rotation degrees for perspective transformation.
        translate (float): Max translation fraction for perspective transformation.
        scale (float): Scale variation range for perspective transformation.
        shear (float): Max shearing factor for perspective transformation.
        perspective (float): Amount of perspective distortion applied.
        post_transform (albumentations.Compose, optional): Additional transformations (e.g., normalization).
        imgsz (int): Target square image size (e.g., 640x640).
        mosaic_prob (float): Probability of applying mosaic augmentation during training.
        is_validation (bool): If True, disables mosaic and all random augmentations.

    Returns:
        A dictionary per item with:
            - 'image': Float tensor of shape (3, imgsz, imgsz), pixel values in [0, 1].
            - 'bboxes': Float tensor of shape (N, 4) in normalized [x1, y1, x2, y2] format.
            - 'labels': Long tensor of shape (N,) with class indices.
    """
    def __init__(self, image_dir, bboxes_dic,
                 degrees=5.0, translate=0.1, scale=0.5, shear=2.0, perspective=0.0001,
                 post_transform=None, 
                 imgsz=640, mosaic_prob=0.5,is_validation = False):
        
        self.image_dir = image_dir
        self.annotations_dict = bboxes_dic
        self.image_names = list(self.annotations_dict.keys())
        self.imgsz = imgsz
        self.mosaic_prob = mosaic_prob
        self.mosaic_border = [-self.imgsz // 2, -self.imgsz // 2]
        self.is_validation = is_validation
        self.random_perspective = RandomPerspective(
            degrees=degrees, translate=translate, scale=scale, shear=shear,
            perspective=perspective, border=self.mosaic_border
        )

    
        self.post_transform = post_transform

        self.letterbox_single = A.Compose([
            A.LongestMaxSize(max_size=self.imgsz, p=1.0),
            A.PadIfNeeded(min_height=self.imgsz, min_width=self.imgsz,
                          border_mode=cv2.BORDER_CONSTANT, fill=(114, 114, 114), p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category']))


        print(f"Initialized DetectionDataset with {len(self.image_names)} images (imgsz={self.imgsz}).")
        print(f" RandomPerspective Params: deg={degrees}, trans={translate}, scale={scale}, shear={shear}, persp={perspective}")
        print(f" Post-Transform: {'Yes' if self.post_transform else 'No'}")


    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, idx):
        if self.is_validation:
            data = self._get_single_image(idx)
            img_raw = data['image']
            bboxes_raw = data['bboxes']
            labels_raw = data['labels']

            # 2. Apply ONLY Letterboxing
            bboxes_list = bboxes_raw.tolist() if len(bboxes_raw)>0 else []
            labels_list = labels_raw.tolist() if len(labels_raw)>0 else []
            letterboxed = self.letterbox_single(image=img_raw,
                                                bboxes=bboxes_list,
                                                category=labels_list)
            img_final = letterboxed['image']   
            bboxes_final = np.array(letterboxed['bboxes'])
            labels_final = np.array(letterboxed['category'])

       
            if len(bboxes_final) > 0:
                    bboxes_final[:, [0, 2]] = bboxes_final[:, [0, 2]].clip(0, self.imgsz)
                    bboxes_final[:, [1, 3]] = bboxes_final[:, [1, 3]].clip(0, self.imgsz)
                    box_w = bboxes_final[:, 2] - bboxes_final[:, 0]; box_h = bboxes_final[:, 3] - bboxes_final[:, 1]
                    keep = (box_w > 1) & (box_h > 1)
                    bboxes_final = bboxes_final[keep]; labels_final = labels_final[keep]


        else:    
            apply_mosaic = random.random() < self.mosaic_prob
            rp_input_labels = None

            if apply_mosaic:
                data = self._get_mosaic(idx)

                instances = Instances(bboxes=data['bboxes'], bbox_format='xyxy', normalized=False)
                
                rp_input_labels = {
                    'img': data['image'], 'cls': data['labels'], 'instances': instances,
                    'mosaic_border': self.mosaic_border
                }
                

            if not apply_mosaic:
                
                data = self._get_single_image(idx)
                img_raw = data['image']           
                bboxes_raw = data['bboxes']      
                labels_raw = data['labels']
                
                letterboxed = self.letterbox_single(image=img_raw,bboxes=bboxes_raw.tolist(),category=labels_raw.tolist())
                img_lb = letterboxed['image']
                bboxes_lb = np.array(letterboxed['bboxes']) 
                labels_lb = np.array(letterboxed['category'])



                instances_lb = Instances(bboxes=bboxes_lb, bbox_format='xyxy', normalized=False)
                rp_input_labels = {
                    'img': img_lb,         
                    'cls': labels_lb,     
                    'instances': instances_lb,
                    'mosaic_border': (0, 0)
                }

                
            if rp_input_labels is None:
                print(f"ERROR: rp_input_labels not set for index {idx}. Returning empty.")
                return {"image": torch.zeros((3, self.imgsz, self.imgsz)), "bboxes": torch.empty((0, 4)), "labels": torch.empty((0,))}

        
    
        
            rp_output_labels = self.random_perspective(rp_input_labels)
            img_final = rp_output_labels['img']          
            instances_after_rp = rp_output_labels['instances']
            labels_final = rp_output_labels['cls']
            bboxes_final = instances_after_rp.bboxes
        
        
            if img_final.shape[0] != self.imgsz or img_final.shape[1] != self.imgsz:
                print(f"Warning: Image shape {img_final.shape} before post_transform is not target size! Resizing.")
                img_final = cv2.resize(img_final, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

                bboxes_final = np.empty((0, 4))
                labels_final = np.array([])

            if self.post_transform:
            
                bboxes_final_list = bboxes_final.tolist() if len(bboxes_final)>0 else []
                labels_final_list = labels_final.tolist() if len(labels_final)>0 else []
                augmented = self.post_transform(
                    image=img_final, bboxes=bboxes_final_list, category=labels_final_list
                )

                img_final = augmented["image"]
                bboxes_final = np.array(augmented["bboxes"])
                labels_final = np.array(augmented["category"])
                
            


        h_norm, w_norm = img_final.shape[:2]
        
        image_tensor = torch.from_numpy(img_final).permute(2, 0, 1).float().div(255.0)
    
        if len(bboxes_final) > 0 and h_norm > 0 and w_norm > 0 :
            bboxes_normalized = bboxes_final.copy()
            bboxes_normalized[:, [0, 2]] /= w_norm
            bboxes_normalized[:, [1, 3]] /= h_norm
            bboxes_normalized = bboxes_normalized.clip(0.0, 1.0)
            bboxes_tensor = torch.from_numpy(bboxes_normalized).float()
            labels_tensor = torch.from_numpy(labels_final).long()
        else:
            bboxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            labels_tensor = torch.empty((0,), dtype=torch.long)


        return {"image": image_tensor, "bboxes": bboxes_tensor, "labels": labels_tensor}


    def _load_image(self, image_name):
        """Loads a single image in RGB format."""
        path = os.path.join(self.image_dir, f"{image_name}.jpg")
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _get_labels(self, image_name):
        """Gets annotations for a single image."""
        annotations = deepcopy(self.annotations_dict[image_name])
       
        bboxes = np.array(annotations["boxes"], dtype=np.float32)
        labels = np.array(annotations["labels"], dtype=np.long)
     
        if len(labels) != len(bboxes):
                print(f"Warning: Mismatch between bbox count ({len(bboxes)}) and label count ({len(labels)}) for {image_name}. Returning empty.")
                bboxes = np.empty((0, 4), dtype=np.float32)
                labels = np.empty((0,), dtype=np.long)

        return {"bboxes": bboxes, "labels": labels} 

    def _get_single_image(self, idx):
        """Loads a single image and its labels, returns raw data."""
        try:
            image_name = self.image_names[idx]
            image = self._load_image(image_name)
            labels_data = self._get_labels(image_name)
       
        except Exception as e:
             print(f"Unexpected error in _get_single_image for index {idx}: {e}. Falling back.")
             return self._get_single_image(np.random.randint(0, len(self)))

        return {
            "image": image,
            "bboxes": labels_data["bboxes"],
            "labels": labels_data["labels"],
            "image_name": image_name,
            "original_shape": image.shape[:2]
        }

    def _get_mosaic(self, idx):
        """Creates a 4-image mosaic, returns large image and labels."""
        mosaic_labels_patches = []
        s = self.imgsz
        canvas_size = s * 2
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)
        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        img_large = np.full((canvas_size, canvas_size, 3), 114, dtype=np.uint8)

        for i, index in enumerate(indices):

            image_name = self.image_names[index]
            img_patch = self._load_image(image_name)
            labels_patch_data = self._get_labels(image_name)
            h, w = img_patch.shape[:2]

            if i == 0:  # tl
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # tr
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, canvas_size), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bl
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(canvas_size, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # br
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, canvas_size), min(canvas_size, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img_large[y1a:y2a, x1a:x2a] = img_patch[y1b:y2b, x1b:x2b]
            padw, padh = x1a - x1b, y1a - y1b

            bboxes_patch = labels_patch_data["bboxes"]
            labels_patch = labels_patch_data["labels"]

            if bboxes_patch.shape[0] > 0:
                bboxes_patch[:, [0, 2]] += padw
                bboxes_patch[:, [1, 3]] += padh

            mosaic_labels_patches.append({
                "bboxes": bboxes_patch,
                "labels": labels_patch,
                "image_name": image_name,
                "original_shape": (h,w)
            })

        final_bboxes, final_labels = self._cat_mosaic_labels(mosaic_labels_patches, canvas_size)

        return {
            "image": img_large,
            "bboxes": final_bboxes, 
            "labels": final_labels,
            "image_name": f"mosaic_{idx}",
            "original_shape": img_large.shape[:2]
        }

    def _cat_mosaic_labels(self, mosaic_labels_patches, canvas_size):
        """Concatenates, clips, and filters labels from mosaic patches."""
        
        if not mosaic_labels_patches: return np.empty((0, 4)), np.empty((0,))
        all_bboxes, all_labels = [], []
        for patch_data in mosaic_labels_patches:
            if patch_data["bboxes"].shape[0] > 0:
                all_bboxes.append(patch_data["bboxes"])
                all_labels.append(patch_data["labels"])
        if not all_bboxes: return np.empty((0, 4)), np.empty((0,))

        final_bboxes = np.concatenate(all_bboxes, axis=0)
        final_labels = np.concatenate(all_labels, axis=0)
        final_bboxes[:, [0, 2]] = final_bboxes[:, [0, 2]].clip(0, canvas_size)
        final_bboxes[:, [1, 3]] = final_bboxes[:, [1, 3]].clip(0, canvas_size)
        box_w, box_h = final_bboxes[:, 2] - final_bboxes[:, 0], final_bboxes[:, 3] - final_bboxes[:, 1]
        keep = (box_w > 1) & (box_h > 1)
        return final_bboxes[keep], final_labels[keep]
  

