import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics.classification import MulticlassJaccardIndex
import numpy as np
from PIL import Image
import argparse
import sys
from transformers import AutoImageProcessor

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import config
from hmr_pt.models.hmr_mask2former import HMRMask2Former
from hmr_pt.datasets.segmentation_dataset import CustomSegmentationDataset, collate_fn
from hmr_pt.losses.hmr_losses import HMRLoss

class HMRPTLightningModule(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        # --- FIX: Use getattr() to safely access the attribute ---
        self.steps_per_epoch = getattr(hparams, "steps_per_epoch", 0)

        self.model = HMRMask2Former(
            model_name=hparams.MODEL_NAME,
            num_classes=hparams.NUM_CLASSES,
            prompt_length_coarse=hparams.PROMPT_LENGTH_COARSE,
            refinement_steps=hparams.REFINEMENT_STEPS,
            prompt_embedding_dim=hparams.PROMPT_EMBEDDING_DIM,
            semantic_init_vlm_model=hparams.SEMANTIC_INIT_VLM_MODEL
        )

        self.task_type = hparams.TASK_TYPE
        if self.task_type == "semantic":
            self.val_mean_iou = MulticlassJaccardIndex(
                num_classes=hparams.NUM_CLASSES,
                ignore_index=255
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
            upsampled_logits = nn.functional.interpolate(
                low_res_logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            predicted_masks = upsampled_logits.argmax(dim=1)
            self.val_mean_iou.update(predicted_masks, labels)

    def on_validation_epoch_end(self):
        if self.task_type == "semantic":
            mean_iou = self.val_mean_iou.compute()
            self.log("val_mean_iou", mean_iou, prog_bar=True, logger=True)

    # --- ADD THIS METHOD ---
    def test_step(self, batch, batch_idx):
        # This is the same logic as validation_step
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = batch["labels"]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels, task_type=self.task_type)
        loss = outputs["loss"]
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        if self.task_type == "semantic":
            low_res_logits = outputs["refined_pred_masks"]
            upsampled_logits = nn.functional.interpolate(
                low_res_logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            predicted_masks = upsampled_logits.argmax(dim=1)
            self.val_mean_iou.update(predicted_masks, labels)

    # --- AND ADD THIS METHOD ---
    def on_test_epoch_end(self):
        # This is the same logic as on_validation_epoch_end
        if self.task_type == "semantic":
            mean_iou = self.val_mean_iou.compute()
            self.log("test_mean_iou", mean_iou, prog_bar=True)


    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        if self.hparams.OPTIMIZER == "AdamW":
            optimizer = optim.AdamW(trainable_params, lr=self.hparams.LEARNING_RATE)
        else:
            optimizer = optim.SGD(trainable_params, lr=self.hparams.LEARNING_RATE)

        if self.steps_per_epoch == 0:
            raise ValueError("steps_per_epoch is not set, which is needed for the scheduler.")

        total_steps = self.steps_per_epoch * self.hparams.NUM_EPOCHS

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.LEARNING_RATE,
            total_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

def main():
    parser = argparse.ArgumentParser(description="HMR-PT Training Script")
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help="Path to the checkpoint file to resume training from (e.g., 'output/last.ckpt')."
    )
    args = parser.parse_args()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    hparams = argparse.Namespace(**{key: getattr(config, key) for key in dir(config) if not key.startswith('__')})

    image_processor_for_datasets = AutoImageProcessor.from_pretrained(hparams.MODEL_NAME)

    if hparams.TASK_TYPE == "semantic":
        train_dataset = CustomSegmentationDataset(
            root_dir=hparams.CITYSCAPES_ROOT,
            image_dir="leftImg8bit/train",
            annotation_dir="gtFine/train",
            image_processor=image_processor_for_datasets,
            task_type=hparams.TASK_TYPE,
            is_train=True
        )
        val_dataset = CustomSegmentationDataset(
            root_dir=hparams.CITYSCAPES_ROOT,
            image_dir="leftImg8bit/val",
            annotation_dir="gtFine/val",
            image_processor=image_processor_for_datasets,
            task_type=hparams.TASK_TYPE,
            is_train=False
        )
    else:
        raise ValueError(f"Unsupported TASK_TYPE: {hparams.TASK_TYPE}")

    train_collate_fn = partial(collate_fn, image_processor=image_processor_for_datasets, task_type=hparams.TASK_TYPE)
    val_collate_fn = partial(collate_fn, image_processor=image_processor_for_datasets, task_type=hparams.TASK_TYPE)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hparams.BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count() // 2,
        collate_fn=train_collate_fn,
        pin_memory=True
    )

    hparams.steps_per_epoch = len(train_dataloader)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hparams.BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        collate_fn=val_collate_fn,
        pin_memory=True
    )

    model_lightning = HMRPTLightningModule(hparams=hparams)

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

    if args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
    else:
        print("Starting training from scratch.")

    print(f"Starting training for {hparams.NUM_EPOCHS} epochs...", flush=True)
    trainer.fit(model_lightning, train_dataloader, val_dataloader, ckpt_path=args.resume_from_checkpoint)

    print("Training complete!", flush=True)

if __name__ == "__main__":
    main()