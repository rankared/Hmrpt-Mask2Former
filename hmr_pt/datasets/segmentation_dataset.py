# hmr_pt/datasets/segmentation_dataset.py

import os
from PIL import Image
import torch
import json
import numpy as np
import albumentations as A

class CustomSegmentationDataset(torch.utils.data.Dataset):
    """
    Custom dataset for Cityscapes with added data augmentation for the training set.
    """
    id_to_train_id = np.array([255, 255, 255, 255, 255, 255, 255, 0, 1, 255, 255, 2, 3, 4, 255, 255, 255, 5, 255, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 255, 255, 16, 17, 18], dtype=np.uint8)

    # --- FIX: Add is_train flag and define augmentations ---
    def __init__(self, root_dir, image_dir, annotation_dir, image_processor, task_type="semantic", is_train=False):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_dir)
        self.annotation_dir = os.path.join(root_dir, annotation_dir)
        self.image_processor = image_processor
        self.task_type = task_type
        self.images = []
        self.annotations = []

        # Define augmentations for the training set only
        if is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.3),
            ])
        else:
            self.transform = None
        # --- END FIX ---

        # (The rest of the file loading logic remains the same)
        if task_type == "semantic" or task_type == "instance":
            for city in os.listdir(self.image_dir):
                city_image_path = os.path.join(self.image_dir, city)
                if not os.path.isdir(city_image_path):
                    continue
                for filename in os.listdir(city_image_path):
                    if filename.endswith("_leftImg8bit.png"):
                        image_path = os.path.join(city_image_path, filename)
                        base_filename = filename.replace("_leftImg8bit.png", "")
                        if task_type == "semantic":
                            annotation_filename = base_filename + "_gtFine_labelIds.png"
                        else: # instance
                            annotation_filename = base_filename + "_gtFine_instanceIds.png"
                        annotation_path = os.path.join(self.annotation_dir, city, annotation_filename)
                        if os.path.exists(annotation_path):
                            self.images.append(image_path)
                            self.annotations.append(annotation_path)
        elif task_type == "panoptic":
            # ... (panoptic loading logic remains unchanged) ...
            pass
        print(f"Loaded {len(self.images)} images for {self.task_type} segmentation.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")

        if self.task_type == "semantic":
            annotation = Image.open(self.annotations[idx])
            
            # --- FIX: Apply augmentations ---
            if self.transform:
                # Convert PIL to numpy for albumentations
                image_np = np.array(image)
                mask_np = np.array(annotation)
                
                transformed = self.transform(image=image_np, mask=mask_np)
                
                # Convert back to PIL Image
                image = Image.fromarray(transformed['image'])
                annotation = Image.fromarray(transformed['mask'])
            # --- END FIX ---

            semantic_map_array = np.array(annotation, dtype=np.uint8)
            remapped_map_array = self.id_to_train_id[semantic_map_array]
            semantic_map = Image.fromarray(remapped_map_array)
            
            inputs = self.image_processor(
                images=image,
                segmentation_maps=semantic_map,
                return_tensors="pt"
            )
            
            inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # (The complex logic to reconstruct 'labels' remains the same)
            if "labels" not in inputs and "mask_labels" in inputs and "class_labels" in inputs:
                mask_labels_list = inputs.pop("mask_labels")
                class_labels_list = inputs.pop("class_labels")
                if not mask_labels_list or not class_labels_list:
                    h, w = inputs["pixel_values"].shape[1:]
                    semantic_map = torch.full((h, w), fill_value=255, dtype=torch.long)
                else:
                    all_masks_tensor = mask_labels_list[0]
                    all_class_ids_tensor = class_labels_list[0]
                    h, w = all_masks_tensor.shape[-2:]
                    semantic_map = torch.full((h, w), fill_value=255, dtype=torch.long)
                    num_masks = all_masks_tensor.shape[0]
                    for i in range(num_masks):
                        mask = all_masks_tensor[i]
                        class_id = all_class_ids_tensor[i]
                        if class_id.item() > 18 and class_id.item() != 255:
                            continue
                        boolean_mask = (mask > 0.5)
                        if boolean_mask.dim() > 2:
                            boolean_mask = boolean_mask.any(dim=0)
                        if boolean_mask.shape == semantic_map.shape:
                            semantic_map[boolean_mask] = class_id.item()
                inputs["labels"] = semantic_map
            return inputs
        
        # (The rest of the __getitem__ for instance/panoptic remains the same)
        elif self.task_type == "instance":
            pass
        elif self.task_type == "panoptic":
            pass

# (The collate_fn remains the same)
def collate_fn(batch, image_processor, task_type="semantic"):
    # ...
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])

    if task_type == "semantic":
        labels = torch.stack([item["labels"] for item in batch])
        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "labels": labels,
        }
    # ...