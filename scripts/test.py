import os
import torch
import lightning as L
from torch.utils.data import DataLoader
from functools import partial
import argparse
import sys

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scripts.config as config
from scripts.train import HMRPTLightningModule
from hmr_pt.datasets.segmentation_dataset import CustomSegmentationDataset, collate_fn
from transformers import AutoImageProcessor

def test():
    parser = argparse.ArgumentParser(description="HMR-PT Testing Script")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the .ckpt file to test (e.g., 'output/semantic/best_model.ckpt')."
    )
    parser.add_argument(
        '--task_type',
        type=str,
        default="semantic",
        help="Task type for the test."
    )
    args = parser.parse_args()

    # --- Load hparams from config ---
    hparams = argparse.Namespace(**{key: getattr(config, key) for key in dir(config) if not key.startswith('__')})
    hparams.TASK_TYPE = args.task_type

    print(f"Loading model from checkpoint: {args.model_path}")

    # --- Load the Lightning Module from checkpoint ---
    # The hparams from config are needed to rebuild the model structure correctly
    model = HMRPTLightningModule.load_from_checkpoint(
        checkpoint_path=args.model_path,
        hparams=hparams
    )

    # --- Create the Test Dataloader (using the validation set) ---
    if hparams.TASK_TYPE == "semantic":
        # Use the model's own image processor that was saved with it
        image_processor = AutoImageProcessor.from_pretrained(model.hparams.MODEL_NAME)
        
        test_dataset = CustomSegmentationDataset(
            root_dir=hparams.CITYSCAPES_ROOT,
            image_dir="leftImg8bit/val",
            annotation_dir="gtFine/val",
            image_processor=image_processor,
            task_type=hparams.TASK_TYPE,
            is_train=False # Ensure no augmentations are used for testing
        )
    else:
        raise ValueError(f"Unsupported TASK_TYPE for testing: {hparams.TASK_TYPE}")
    
    test_collate_fn = partial(collate_fn, image_processor=image_processor, task_type=hparams.TASK_TYPE)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=hparams.BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        collate_fn=test_collate_fn,
        pin_memory=True
    )
    
    # --- Setup Trainer and Run Test ---
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False # No need to log during testing
    )
    
    print("Starting evaluation on the validation set...")
    trainer.test(model, dataloaders=test_dataloader)
    print("Evaluation complete!")


if __name__ == "__main__":
    test()