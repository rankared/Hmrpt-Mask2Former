import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import evaluate # For evaluation metrics [1]
import numpy as np
from PIL import Image

# Ensure the current script's directory is in sys.path to prioritize your config.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import configuration and custom modules
import config # Your custom config.py [1]
from hmr_pt.models.hmr_mask2former import HMRMask2Former
from hmr_pt.datasets.segmentation_dataset import CustomSegmentationDataset, collate_fn
from hmr_pt.losses.hmr_losses import HMRLoss # Your custom loss [1]

# --- PyTorch Lightning Module for HMR-PT ---
class HMRPTLightningModule(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams) # Save all hyperparameters from config

        # Initialize HMRMask2Former model
        self.model = HMRMask2Former(
            model_name=hparams.MODEL_NAME,
            num_classes=hparams.NUM_CLASSES,
            prompt_length_coarse=hparams.PROMPT_LENGTH_COARSE,
            prompt_length_fine=hparams.PROMPT_LENGTH_FINE,
            num_decoder_layers=6, # Mask2Former typically has 6 decoder layers [2]
            refinement_steps=hparams.REFINEMENT_STEPS,
            prompt_embedding_dim=hparams.PROMPT_EMBEDDING_DIM,
            semantic_init_vlm_model=hparams.SEMANTIC_INIT_VLM_MODEL
        )

        # Initialize evaluation metrics
        # For semantic segmentation, mIoU is standard [3]
        self.metric = evaluate.load("mean_iou") # [1]

    def forward(self, pixel_values, pixel_mask):
        # The forward method of the LightningModule calls the forward of your HMRMask2Former
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = batch["labels"] # Ground truth segmentation maps

        # Forward pass through HMR-PT model
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        
        loss = outputs["loss"]
        
        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = batch["labels"]

        # Forward pass
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        
        loss = outputs["loss"]
        
        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # --- Calculate Metrics for Refined Masks ---
        # Mask2Former outputs logits, need to convert to predictions
        # refined_pred_masks: (batch_size, num_classes, H, W)
        # labels: (batch_size, H, W) - class IDs
        
        # Resize labels to match prediction size if necessary
        # Mask2Former's processor usually handles this, but double-check
        target_masks_resized = torch.nn.functional.interpolate(
            labels.unsqueeze(1).float(),
            size=outputs["refined_pred_masks"].shape[-2:],
            mode='nearest'
        ).squeeze(1).long()

        # Get predicted class IDs (argmax over class dimension)
        # For semantic segmentation, take argmax over class dimension
        predicted_masks = outputs["refined_pred_masks"].argmax(dim=1) # (batch_size, H, W)

        # Update evaluation metric
        # The evaluate.load("mean_iou") expects predictions and references as lists of numpy arrays
        # or PIL images.
        # Ensure to handle ignore_index (e.g., 255 for Cityscapes unlabeled)
        
        # Convert tensors to numpy arrays for evaluation library
        # Exclude ignore_index from evaluation if present
        
        # For evaluate.load("mean_iou"), you typically pass lists of predictions and references
        # and specify ignore_index.
        
        # Example for semantic segmentation:
        # Flatten predictions and labels, filter out ignore_index
        predictions_flat = predicted_masks.view(-1)
        labels_flat = target_masks_resized.view(-1)

        # Filter out ignored pixels (e.g., 255 for Cityscapes)
        valid_pixels = (labels_flat!= 255) # Assuming 255 is ignore_index
        
        # Ensure predictions are within valid class range for metric calculation
        # Some models might predict outside valid class IDs for ignored regions
        # Clamp predictions to valid class range if necessary
        predictions_flat_valid = predictions_flat[valid_pixels]
        labels_flat_valid = labels_flat[valid_pixels]

        # Convert to numpy for evaluate library
        self.metric.add_batch(
            predictions=predictions_flat_valid.cpu().numpy(),
            references=labels_flat_valid.cpu().numpy()
        )

    def on_validation_epoch_end(self):
        # Compute and log epoch-level metrics
        metrics = self.metric.compute(
            num_labels=self.hparams.NUM_CLASSES,
            ignore_index=255, # Specify ignore index for Cityscapes [3]
            # Adjust average="weighted" or "macro" based on your needs
            # For mIoU, it's usually "macro" or "none" to get per-class then average
        )
        
        # Log mean_iou and pixel_accuracy
        mean_iou = metrics["mean_iou"]
        pixel_accuracy = metrics["pixel_accuracy"]
        
        self.log("val_mean_iou", mean_iou, prog_bar=True, logger=True)
        self.log("val_pixel_accuracy", pixel_accuracy, prog_bar=True, logger=True)
        
        # If you want to log per-class IoU, you can access metrics["per_category_iou"]
        # and log them individually or as a dictionary.
        
        # Reset metric for next epoch
        self.metric.empty()

    def configure_optimizers(self):
        # Define optimizer
        if self.hparams.OPTIMIZER == "AdamW":
            optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams.LEARNING_RATE)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=self.hparams.LEARNING_RATE)
        return optimizer

# --- Main Training Function ---
def main():
    # Create output directory if it doesn't exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Initialize the Lightning Module with hyperparameters from config
    model_lightning = HMRPTLightningModule(hparams=config)

    # Prepare datasets and dataloaders
    # The image_processor is part of the HMRMask2Former model
    image_processor = model_lightning.model.processor

    # For Cityscapes dataset
    if config.TASK_TYPE == "semantic": # [1]
        train_dataset = CustomSegmentationDataset(
            root_dir=config.CITYSCAPES_ROOT,
            image_dir="leftImg8bit/train", # Adjust based on your extracted structure
            annotation_dir="gtFine/train", # Adjust based on your extracted structure
            image_processor=image_processor,
            task_type=config.TASK_TYPE
        )
        val_dataset = CustomSegmentationDataset(
            root_dir=config.CITYSCAPES_ROOT,
            image_dir="leftImg8bit/val",
            annotation_dir="gtFine/val",
            image_processor=image_processor,
            task_type=config.TASK_TYPE
        )
    elif config.TASK_TYPE == "instance" or config.TASK_TYPE == "panoptic":
        # For COCO or other datasets, you'd load them here.
        # This would require a CustomSegmentationDataset that handles COCO JSON format [4]
        # For example:
        # train_dataset = CustomSegmentationDataset(
        #     root_dir=config.COCO_ROOT,
        #     image_dir="train2017",
        #     annotation_dir="annotations/instances_train2017.json", # Or panoptic_train2017.json
        #     image_processor=image_processor,
        #     task_type=config.TASK_TYPE
        # )
        # val_dataset = CustomSegmentationDataset(
        #     root_dir=config.COCO_ROOT,
        #     image_dir="val2017",
        #     annotation_dir="annotations/instances_val2017.json",
        #     image_processor=image_processor,
        #     task_type=config.TASK_TYPE
        # )
        raise NotImplementedError(f"Dataset loading for {config.TASK_TYPE} is not yet implemented. Please implement it in segmentation_dataset.py.")
    else:
        raise ValueError(f"Unsupported TASK_TYPE: {config.TASK_TYPE}")

    # Create data loaders,
    # Use partial to pass image_processor to collate_fn
    train_collate_fn = partial(collate_fn, image_processor=image_processor)
    val_collate_fn = partial(collate_fn, image_processor=image_processor)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count() // 2, # Use half of CPU cores for data loading
        collate_fn=train_collate_fn,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        collate_fn=val_collate_fn,
        pin_memory=True
    )

    # --- PyTorch Lightning Trainer Setup ---
    # Logger for saving training progress to CSV,
    csv_logger = CSVLogger(save_dir=config.OUTPUT_DIR, name="hmr_pt_logs")

    # Model checkpointing callback,
    # Saves the best model based on validation mean_iou
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.OUTPUT_DIR,
        filename="best_hmr_pt_model-{epoch:02d}-{val_mean_iou:.4f}",
        monitor="val_mean_iou", # Metric to monitor for saving best model
        mode="max", # Save model when val_mean_iou is maximized
        save_top_k=1, # Save only the top 1 best model
        save_last=True # Also save the last epoch's model
    )

    # Initialize the Lightning Trainer
    trainer = L.Trainer(
        max_epochs=config.NUM_EPOCHS,
        logger=csv_logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu", # Use GPU if available
        devices=1, # Number of GPUs to use (adjust as per your VM setup)
        # precision="16-mixed", # Use mixed precision for faster training if supported by your GPU
        # If you encounter issues with precision, comment this out or use "bf16-mixed"
        log_every_n_steps=50, # Log every N steps
        val_check_interval=0.5, # Run validation every 0.5 epoch
        # For two-stage training, you might need to manage ckpt_path for resuming
        # ckpt_path=config.CHECKPOINT_PATH_STAGE1 if config.TWO_STAGE_TRAINING else None
    )

    # --- Training Loop ---
    print(f"Starting training for {config.NUM_EPOCHS} epochs...")
    trainer.fit(model_lightning, train_dataloader, val_dataloader)
    print("Training complete!")

    # --- Optional: Two-Stage Training Logic (Conceptual) ---
    # If config.TWO_STAGE_TRAINING is True, you would typically:
    # 1. Train Stage 1 (e.g., only prompts, or specific modules)
    #    trainer.fit(model_lightning, train_dataloader, val_dataloader)
    #    # Save checkpoint after stage 1
    #    trainer.save_checkpoint(config.OUTPUT_DIR + "stage1_model.ckpt")
    #
    # 2. Load Stage 1 checkpoint, potentially unfreeze more layers/modules
    #    model_lightning_stage2 = HMRPTLightningModule.load_from_checkpoint(
    #        config.OUTPUT_DIR + "stage1_model.ckpt",
    #        hparams=config # Pass hparams again
    #    )
    #    # Adjust optimizer for stage 2 (e.g., different learning rate, unfreeze more params)
    #    # trainer.fit(model_lightning_stage2, train_dataloader, val_dataloader, ckpt_path=config.OUTPUT_DIR + "stage1_model.ckpt")
    #    # This would require more complex logic in configure_optimizers or separate trainers.

if __name__ == "__main__":
    main()
