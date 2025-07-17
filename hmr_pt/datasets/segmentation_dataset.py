import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from transformers import AutoImageProcessor # For Mask2Former's image processing [3]

class CustomSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_dir, annotation_dir, image_processor, transform=None, task_type="semantic"):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_dir)
        self.annotation_dir = os.path.join(root_dir, annotation_dir)
        self.image_processor = image_processor
        self.transform = transform
        self.task_type = task_type
        
        self.image_paths = sorted([os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.annotation_paths = sorted([os.path.join(self.annotation_dir, f) for f in os.listdir(self.annotation_dir) if f.endswith('.png')]) # Assuming mask images

        # For Cityscapes, annotations are typically PNGs where pixel values map to class IDs [3]
        # For instance/panoptic, COCO format JSONs are common [7, 8]
        
        # Example: For Cityscapes semantic segmentation, you'd load the semantic mask image
        # For instance/panoptic, you'd parse COCO JSON to get masks and class IDs.
        # This example assumes semantic segmentation with mask images.

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        ann_path = self.annotation_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(ann_path).convert("L") # Load as grayscale for class IDs

        # Apply transformations (e.g., resizing, normalization)
        if self.transform:
            image, mask = self.transform(image, mask) # Custom transform that handles both

        # Mask2Former's image processor handles resizing, normalization, etc.
        # It expects a list of images and a list of annotations (e.g., dicts for panoptic)
        # For semantic segmentation, it might expect a single mask tensor.
        
        # This part needs to align with Mask2Former's expected input format for labels
        # For semantic segmentation, labels are typically (H, W) tensor of class IDs.
        # The processor will handle image normalization and batching.
        
        # Example for semantic segmentation:
        inputs = self.image_processor(images=image, segmentation_maps=mask, return_tensors="pt")
        
        # The pixel_values and pixel_mask are ready for the model
        # The labels (segmentation_maps) are also processed by image_processor
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0), # Remove batch dim for single image
            "pixel_mask": inputs["pixel_mask"].squeeze(0),
            "labels": inputs["labels"].squeeze(0) # This is the processed segmentation map
        }

# Collate function for DataLoader (if needed, depending on processor usage)
def collate_fn(batch, image_processor):
    pixel_values = [item["pixel_values"] for item in batch]
    pixel_mask = [item["pixel_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Use image_processor's batching capabilities if it supports it for labels
    # Otherwise, manually pad/stack
    # For Mask2Former, the processor often handles this during __getitem__
    # So, here we just stack the already processed tensors.
    
    return {
        "pixel_values": torch.stack(pixel_values),
        "pixel_mask": torch.stack(pixel_mask),
        "labels": torch.stack(labels)
        }
