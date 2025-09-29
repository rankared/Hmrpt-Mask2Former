# hmr_pt/datasets/segmentation_dataset.py

import os
from PIL import Image
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
import json

class CustomSegmentationDataset(Dataset):
    """
    Generalized custom dataset for Cityscapes, supporting "semantic", "instance",
    and "panoptic" segmentation tasks. It correctly processes data for each
    task type as expected by the Mask2Former model.
    """
    # Official Cityscapes mapping from original IDs to 19 training classes + ignore class
    id_to_train_id = np.array([255, 255, 255, 255, 255, 255, 255, 0, 1, 255, 255, 2, 3, 4, 255, 255, 255, 5, 255, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 255, 255, 16, 17, 18], dtype=np.uint8)

    def __init__(self, root_dir, image_dir, annotation_dir, image_processor, task_type="semantic", is_train=False):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_dir)
        self.annotation_dir = os.path.join(root_dir, annotation_dir)
        self.image_processor = image_processor
        self.task_type = task_type
        self.is_train = is_train
        self.images = []
        self.annotations = []

        # Define augmentations for the training set only
        if self.is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.3),
            ])
        else:
            self.transform = None

        # Load file paths based on the specified task type
        for city in os.listdir(self.image_dir):
            city_image_path = os.path.join(self.image_dir, city)
            if not os.path.isdir(city_image_path):
                continue
            
            for filename in os.listdir(city_image_path):
                if filename.endswith("_leftImg8bit.png"):
                    image_path = os.path.join(city_image_path, filename)
                    base_filename = filename.replace("_leftImg8bit.png", "")
                    
                    annotation_path = None
                    if self.task_type == "semantic":
                        annotation_filename = base_filename + "_gtFine_labelIds.png"
                        annotation_path = os.path.join(self.annotation_dir, city, annotation_filename)
                    elif self.task_type == "instance":
                        annotation_filename = base_filename + "_gtFine_instanceIds.png"
                        annotation_path = os.path.join(self.annotation_dir, city, annotation_filename)
                    elif self.task_type == "panoptic":
                        # For panoptic, we need both the PNG and the JSON file
                        panoptic_png_filename = base_filename + "_gtFine_panoptic.png"
                        panoptic_json_filename = base_filename + "_gtFine_polygons.json"
                        png_path = os.path.join(self.annotation_dir.replace("gtFine", "gtFine_panoptic"), city, panoptic_png_filename)
                        json_path = os.path.join(self.annotation_dir, city, panoptic_json_filename)
                        if os.path.exists(png_path) and os.path.exists(json_path):
                            annotation_path = (png_path, json_path) # Store as a tuple
                    
                    if annotation_path and (isinstance(annotation_path, str) and os.path.exists(annotation_path) or isinstance(annotation_path, tuple)):
                        self.images.append(image_path)
                        self.annotations.append(annotation_path)

        print(f"Loaded {len(self.images)} images for '{self.task_type}' segmentation ({'train' if is_train else 'val'}).")

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        annotation = Image.open(self.annotations[idx])

        # Convert original annotation to numpy for later use
        original_annotation_np = np.array(annotation, dtype=np.uint8)

        inputs = self.image_processor(
            images=image,
            segmentation_maps=annotation,
            return_tensors="pt",
            return_segmentation_maps=True,
        )

        processed_inputs = {
            k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        if self.task_type == "semantic" and "class_labels" in processed_inputs:
            mask_labels = processed_inputs["mask_labels"]
            
            if len(mask_labels) > 0:
                mask_shape = mask_labels[0].shape
                h, w = mask_shape[-2], mask_shape[-1]
                
                # --- THE DEFINITIVE FIX ---
                # Create the final semantic map from the original annotation,
                # but resized to match the processor's output dimensions.
                
                # 1. Remap the original full-size annotation to train_ids
                remapped_annotation = self.id_to_train_id[original_annotation_np]
                
                # 2. Convert to a tensor and add a channel dim for resizing
                remapped_tensor = torch.from_numpy(remapped_annotation).unsqueeze(0).float()
                
                # 3. Resize it to the exact same size as the model's masks (h, w)
                #    using 'nearest' interpolation to preserve label integrity.
                resized_semantic_map = torch.nn.functional.interpolate(
                    remapped_tensor.unsqueeze(0), 
                    size=(h, w), 
                    mode='nearest'
                ).squeeze(0).squeeze(0)
                
                # 4. Assign the correctly processed map and clean up
                processed_inputs["labels"] = resized_semantic_map.long()
                del processed_inputs["mask_labels"]
                del processed_inputs["class_labels"]

        return processed_inputs

def collate_fn(batch, image_processor):
    """
    Generalized collate function that correctly routes semantic and
    instance/panoptic data to the appropriate padding/stacking logic.
    """
    # Check a flag to print the trace only for the first few batches
    if not hasattr(collate_fn, "trace_done"):
        collate_fn.trace_count = 0
        collate_fn.trace_done = False

    # The key 'labels' containing a single tensor is the definitive sign of a semantic task.
    if "labels" in batch[0] and isinstance(batch[0]["labels"], torch.Tensor):
        # --- SEMANTIC PATH: Manually stack the tensors ---
        #if not collate_fn.trace_done:
        #    print("\n--- [TRACE] collate_fn: Detected semantic task. Using torch.stack() path. ---\n")
        
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        
        if "pixel_mask" in batch[0] and batch[0]["pixel_mask"] is not None:
            pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
        else:
            pixel_mask = torch.ones_like(pixel_values)
            
        labels = torch.stack([item["labels"] for item in batch])
        
        if not collate_fn.trace_done:
            collate_fn.trace_count += 1
            if collate_fn.trace_count > 2: # Print for the first 3 batches
                collate_fn.trace_done = True
        
        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "labels": labels,
        }
    else:
        # --- INSTANCE / PANOPTIC PATH: Use the processor's powerful pad function ---
        if not collate_fn.trace_done:
            print("\n--- [TRACE] collate_fn: Detected instance/panoptic task. Using image_processor.pad() path. ---\n")
            collate_fn.trace_count += 1
            if collate_fn.trace_count > 2:
                collate_fn.trace_done = True
                
        return image_processor.pad(batch, return_tensors="pt")

