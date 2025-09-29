import os
import torch
from torch.utils.data import DataLoader
from functools import partial
import argparse
import sys
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import json
import logging
import numpy as np
from PIL import Image

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scripts.config as config
from hmr_pt.datasets.segmentation_dataset import CustomSegmentationDataset, collate_fn

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to save visual results for qualitative analysis
def save_visual_examples(batch, predictions, processor, output_dir, batch_idx, num_examples=2): # <-- Pass processor in
    """Saves a few visual examples from a batch."""
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    color_map = config.LABEL_COLORS_CITYSCAPES
    # Use the passed-in processor's mean and std
    mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std).view(1, 3, 1, 1)

    for i in range(min(num_examples, len(batch["pixel_values"]))):
        # Denormalize image for viewing
        img_tensor = batch["pixel_values"][i].cpu() * std + mean
        gt_mask = batch["labels"][i].cpu().numpy()
        
        # Denormalize and resize image to match ground truth for visualization
        img_numpy = (img_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_numpy)
        target_size = (gt_mask.shape[1], gt_mask.shape[0]) # (width, height)
        img_tensor_resized = np.array(img_pil.resize(target_size))

        # Convert label and prediction masks to color images
        pred_mask = predictions[i].cpu().numpy()
        gt_colored = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        pred_colored = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)

        for label_id, color in enumerate(color_map):
            gt_colored[gt_mask == label_id] = color
            pred_colored[pred_mask == label_id] = color
        
        # Concatenate [Original | Ground Truth | Prediction] for comparison
        concatenated_image = Image.fromarray(np.concatenate([
            img_tensor_resized, 
            gt_colored, 
            pred_colored
        ], axis=1))

        concatenated_image.save(os.path.join(vis_dir, f"batch_{batch_idx}_example_{i}.png"))

@torch.no_grad()
def test_baseline():
    parser = argparse.ArgumentParser(description="Baseline Mask2Former Testing Script")
    parser.add_argument('--model_name', type=str, required=True, help="Hugging Face model identifier.")
    parser.add_argument('--num_vis_batches', type=int, default=1, help="Number of batches to save for visual inspection.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    logging.info(f"Loading pre-trained model: {args.model_name}")
    
    # --- CRITICAL FIX HERE ---
    # Configure the main processor with the correct size to prevent aspect ratio distortion.
    image_processor = AutoImageProcessor.from_pretrained(
        args.model_name, 
        size={"height": 1024, "width": 2048}
    )
    
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model_name).to(device)
    model.eval()

    hparams = argparse.Namespace(**{key: getattr(config, key) for key in dir(config) if not key.startswith('__')})
    
    # This dataset will now use the correctly configured processor
    test_dataset = CustomSegmentationDataset(
        root_dir=hparams.CITYSCAPES_ROOT, image_dir="leftImg8bit/val", annotation_dir="gtFine/val",
        image_processor=image_processor, task_type="semantic", is_train=False
    )
        
    test_collate_fn = partial(collate_fn, image_processor=image_processor)
    test_dataloader = DataLoader(
        test_dataset, batch_size=hparams.BATCH_SIZE, shuffle=False,
        num_workers=os.cpu_count() // 2, collate_fn=test_collate_fn, pin_memory=True
    )

    logging.info("Starting evaluation on the validation set...")
    mean_iou_metric = MulticlassJaccardIndex(num_classes=hparams.NUM_CLASSES, ignore_index=255).to(device)
    
    for i, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        target_sizes = [labels.shape[-2:] for _ in range(len(pixel_values))]

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            outputs = model(pixel_values=pixel_values)

        predicted_masks_list = image_processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        predicted_masks = torch.stack(predicted_masks_list).to(device)
        
        mean_iou_metric.update(predicted_masks, labels)

        if i < args.num_vis_batches:
            # Pass the correctly configured processor to the visualization function
            save_visual_examples(batch, predicted_masks, image_processor, hparams.OUTPUT_DIR, batch_idx=i)

    final_miou = mean_iou_metric.compute().item()
    
    results = {
        "model_name": args.model_name,
        "dataset": "Cityscapes Validation",
        "mIoU": final_miou,
        "mIoU_percent": f"{final_miou*100:.2f}%"
    }
    results_path = os.path.join(hparams.OUTPUT_DIR, "evaluation_results.json")
    os.makedirs(hparams.OUTPUT_DIR, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info("\n" + "="*50)
    logging.info(f"Evaluation Complete!")
    logging.info(f"Baseline Model: {args.model_name}")
    logging.info(f"Final Validation mIoU: {final_miou:.4f} ({final_miou*100:.2f}%)")
    logging.info(f"Results saved to {results_path}")
    logging.info("="*50)

if __name__ == "__main__":
    test_baseline()