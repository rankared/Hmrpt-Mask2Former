import pytorch_lightning as pl
import torch
import config

torch.manual_seed(1)
torch.set_float32_matmul_precision("medium")
from mask2former import ( Mask2FormerFinetuner, 
                        SegmentationDataModule, 
                        DATASET_DIR, 
                        BATCH_SIZE, 
                        NUM_WORKERS, 
                        ID2LABEL, 
                        LEARNING_RATE, 
                        LOGGER,  
                        DEVICES,
                        CHECKPOINT_CALLBACK, 
                        EPOCHS )


if __name__=="__main__":
    data_module = SegmentationDataModule(dataset_dir=DATASET_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
#    model=Mask2FormerFinetuner(ID2LABEL, LEARNING_RATE)

    model = HMRMask2Former(
      model_name=config.MODEL_NAME,
      num_classes=config.NUM_CLASSES,
      prompt_length_coarse=config.PROMPT_LENGTH_COARSE,
      prompt_length_fine=config.PROMPT_LENGTH_FINE,
      refinement_steps=config.REFINEMENT_STEPS,
      prompt_embedding_dim=config.PROMPT_EMBEDDING_DIM,
      semantic_init_vlm_model=config.SEMANTIC_INIT_VLM_MODEL
    )

    train_dataset = CustomSegmentationDataset(
      root_dir=config.CITYSCAPES_ROOT, # Or ADE20K_ROOT, COCO_ROOT based on which dataset you're using
      image_dir="leftImg8bit", # Example for Cityscapes
      annotation_dir="gtFine", # Example for Cityscapes
      image_processor=model.processor,
      task_type=config.TASK_TYPE
    )

    trainer = pl.Trainer(
        logger=LOGGER,
        accelerator='cuda',
        devices=DEVICES,
        strategy="ddp",
        callbacks=[CHECKPOINT_CALLBACK],
        max_epochs=EPOCHS
    )
    print("Training starts!!")
    trainer.fit(model,data_module)
    print("saving model!")
    trainer.save_checkpoint("mask2former.ckpt")
