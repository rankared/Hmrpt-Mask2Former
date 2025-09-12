import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
#  Swapping to a more reliable metrics library
from torchmetrics.classification import MulticlassJaccardIndex
import numpy as np
from PIL import Image
import argparse
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import config
from hmr_pt.models.hmr_mask2former import HMRMask2Former
from hmr_pt.datasets.segmentation_dataset import CustomSegmentationDataset, collate_fn
from hmr_pt.losses.hmr_losses import HMRLoss

class HMRPTLightningModule(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams) 

        self.model = HMRMask2Former(
            model_name=hparams.MODEL_NAME,
            num_classes=hparams.NUM_CLASSES,
            prompt_length_coarse=hparams.PROMPT_LENGTH_COARSE,
            prompt_length_fine=hparams.PROMPT_LENGTH_FINE,
            refinement_steps=hparams.REFINEMENT_STEPS,
            prompt_embedding_dim=hparams.PROMPT_EMBEDDING_DIM,
            semantic_init_vlm_model=hparams.SEMANTIC_INIT_VLM_MODEL
        )

        self.task_type = hparams.TASK_TYPE
        if self.task_type == "semantic":
            # 🎯 DEFINITIVE FIX: Use torchmetrics for stable and direct tensor-based metric calculation.
            # MulticlassJaccardIndex is the official name for Mean IoU.
            self.val_mean_iou = MulticlassJaccardIndex(
                num_classes=hparams.NUM_CLASSES, 
                ignore_index=255 # Cityscapes ignore index
            )
        else:
            raise ValueError(f"Unsupported TASK_TYPE for evaluation: {self.task_type}")

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = batch["labels"]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels, task_type=self.task_type)
        
        loss = outputs["loss"]
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = batch["labels"]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels, task_type=self.task_type)

        loss = outputs["loss"]
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.task_type == "semantic":
            low_res_logits = outputs["refined_pred_masks"]

            # Upsample predicted logits to match the label's spatial dimensions.
            upsampled_logits = nn.functional.interpolate(
                low_res_logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            predicted_masks = upsampled_logits.argmax(dim=1)

            # 🎯 Update metric directly with tensors. No more conversions needed.
            self.val_mean_iou.update(predicted_masks, labels)

    def on_validation_epoch_end(self):
        if self.task_type == "semantic":
            # 🎯 Compute and log the metric from torchmetrics.
            mean_iou = self.val_mean_iou.compute()
            self.log("val_mean_iou", mean_iou, prog_bar=True, logger=True)
            
            # The metric state is automatically reset at the start of the next epoch.

    def configure_optimizers(self):
        # Filter parameters to only optimize those that require gradients
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.hparams.OPTIMIZER == "AdamW":
            optimizer = optim.AdamW(trainable_params, lr=self.hparams.LEARNING_RATE)
        else:
            optimizer = optim.SGD(trainable_params, lr=self.hparams.LEARNING_RATE)
        return optimizer

def main():
    # --- NEW CODE START: Add ArgumentParser ---
    parser = argparse.ArgumentParser(description="HMR-PT Training Script")
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help="Path to the checkpoint file to resume training from (e.g., 'output/last.ckpt')."
    )
    args = parser.parse_args()
    # --- NEW CODE END ---

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    hparams = argparse.Namespace(**{key: getattr(config, key) for key in dir(config) if not key.startswith('__')})

    model_lightning = HMRPTLightningModule(hparams=hparams)
    image_processor = model_lightning.model.processor

    if hparams.TASK_TYPE == "semantic":
        train_dataset = CustomSegmentationDataset(
            root_dir=hparams.CITYSCAPES_ROOT,
            image_dir="leftImg8bit/train",
            annotation_dir="gtFine/train",
            image_processor=image_processor,
            task_type=hparams.TASK_TYPE
        )
        val_dataset = CustomSegmentationDataset(
            root_dir=hparams.CITYSCAPES_ROOT,
            image_dir="leftImg8bit/val",
            annotation_dir="gtFine/val",
            image_processor=image_processor,
            task_type=hparams.TASK_TYPE
        )
    else:
        raise ValueError(f"Unsupported TASK_TYPE: {hparams.TASK_TYPE}")

    train_collate_fn = partial(collate_fn, image_processor=image_processor, task_type=hparams.TASK_TYPE)
    val_collate_fn = partial(collate_fn, image_processor=image_processor, task_type=hparams.TASK_TYPE)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hparams.BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count() // 2,
        collate_fn=train_collate_fn,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hparams.BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        collate_fn=val_collate_fn,
        pin_memory=True
    )

    csv_logger = CSVLogger(save_dir=hparams.OUTPUT_DIR, name="hmr_pt_logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.OUTPUT_DIR,
        filename="best_hmr_pt_model-{epoch:02d}-{val_mean_iou:.4f}",
        monitor="val_mean_iou",
        mode="max",
        save_top_k=1,
        save_last=True
    )

    trainer = L.Trainer(
        max_epochs=hparams.NUM_EPOCHS,
        logger=csv_logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=1,
        log_every_n_steps=50,
        val_check_interval=1.0,
    )
  
    # --- NEW CODE START: Add conditional print statements ---
    if args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
    else:
        print("Starting training from scratch.")
    # --- NEW CODE END ---

    print(f"Starting training for {hparams.NUM_EPOCHS} epochs...", flush=True)
    #trainer.fit(model_lightning, train_dataloader, val_dataloader)
    # --- MODIFIED LINE: Pass the checkpoint path to trainer.fit() ---
    trainer.fit(model_lightning, train_dataloader, val_dataloader, ckpt_path=args.resume_from_checkpoint)

    print("Training complete!", flush=True)

if __name__ == "__main__":
    main()