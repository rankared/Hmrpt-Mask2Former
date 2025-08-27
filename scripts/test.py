import pytorch_lightning as pl
import torch
torch.manual_seed(1)
torch.set_float32_matmul_precision("medium")

# Import from the current project structure
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import config # For HMR-PT specific configs like TASK_TYPE
from train import HMRPTLightningModule # Import the Lightning Module
from hmr_pt.datasets.segmentation_dataset import CustomSegmentationDataset, collate_fn
from torch.utils.data import DataLoader
from functools import partial
import argparse

# Assuming config.py defines these
from mask2former import ( DATASET_DIR, BATCH_SIZE, NUM_WORKERS, ID2LABEL, LEARNING_RATE, LOGGER, PRECISION)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default='',
        help="Enter the path of your model.ckpt file"
    )
    parser.add_argument(
        '--task_type',
        type=str,
        default='semantic', # Default to semantic
        choices=['semantic', 'instance', 'panoptic'],
        help="Specify the segmentation task type: 'semantic', 'instance', or 'panoptic'"
    )
    parser.add_argument(
        '--cityscapes_root',
        type=str,
        default='./data/cityscapes', # Adjust as per your Cityscapes data path
        help="Root directory of the Cityscapes dataset"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=config.BATCH_SIZE if hasattr(config, 'BATCH_SIZE') else 1,
        help="Batch size for testing"
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=os.cpu_count() // 2,
        help="Number of data loader workers"
    )

    args = parser.parse_args()
    model_path = args.model_path
    task_type = args.task_type
    cityscapes_root = args.cityscapes_root
    batch_size = args.batch_size
    num_workers = args.num_workers

    # Create a Namespace object for hparams, similar to how it's done in train.py
    # This is crucial because HMRPTLightningModule expects a Namespace with attributes
    # like NUM_CLASSES, TASK_TYPE, etc.
    hparams = argparse.Namespace(**{key: getattr(config, key) for key in dir(config) if not key.startswith('__')})
    hparams.TASK_TYPE = task_type # Override with CLI argument if provided
    hparams.CITYSCAPES_ROOT = cityscapes_root
    hparams.BATCH_SIZE = batch_size
    hparams.NUM_WORKERS = num_workers

    # Data Module for testing
    # We'll use CustomSegmentationDataset directly and DataLoader

    if hparams.TASK_TYPE == "semantic":
        test_dataset = CustomSegmentationDataset(
            root_dir=hparams.CITYSCAPES_ROOT,
            image_dir="leftImg8bit/val", # Use validation set as test set in this example
            annotation_dir="gtFine/val",
            image_processor=HMRPTLightningModule.load_from_checkpoint(model_path, hparams=hparams).model.processor, # Load processor from model
            task_type=hparams.TASK_TYPE
        )
    elif hparams.TASK_TYPE == "instance":
        test_dataset = CustomSegmentationDataset(
            root_dir=hparams.CITYSCAPES_ROOT,
            image_dir="leftImg8bit/val",
            annotation_dir="gtFine/val", # gtFine for instance
            image_processor=HMRPTLightningModule.load_from_checkpoint(model_path, hparams=hparams).model.processor,
            task_type=hparams.TASK_type
        )
    elif hparams.TASK_TYPE == "panoptic":
        test_dataset = CustomSegmentationDataset(
            root_dir=hparams.CITYSCAPES_ROOT,
            image_dir="leftImg8bit/val",
            annotation_dir="gtPanoptic/val", # gtPanoptic for panoptic
            image_processor=HMRPTLightningModule.load_from_checkpoint(model_path, hparams=hparams).model.processor,
            task_type=hparams.TASK_TYPE
        )
    else:
        raise ValueError(f"Unsupported TASK_TYPE: {hparams.TASK_TYPE}")

    test_collate_fn = partial(collate_fn, image_processor=test_dataset.image_processor, task_type=hparams.TASK_TYPE)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=hparams.BATCH_SIZE,
        shuffle=False,
        num_workers=hparams.NUM_WORKERS,
        collate_fn=test_collate_fn,
        pin_memory=True
    )

    # Load the HMRPTLightningModule from the checkpoint
    # It's important to pass the hparams to ensure model reconstruction
    model = HMRPTLightningModule.load_from_checkpoint(model_path, hparams=hparams)

    # Configure trainer logger and precision based on config
    trainer_logger = pl.loggers.CSVLogger(save_dir=config.OUTPUT_DIR, name="hmr_pt_test_logs") if hasattr(config, 'OUTPUT_DIR') else None
    trainer_precision = config.PRECISION if hasattr(config, 'PRECISION') else "32-true"

    trainer = pl.Trainer(
        logger=trainer_logger,
        precision=trainer_precision,
        accelerator='cuda',
        devices=[0],
        num_nodes=1,
    )
    print(f"Test starts for {hparams.TASK_TYPE} task!!")
    trainer.test(model, test_dataloader)
