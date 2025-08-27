# config.py

# --- General Training Configuration ---
# Set the task type for segmentation: "semantic", "instance", or "panoptic"
TASK_TYPE = "semantic" # [1, 2, 3]

# Number of training epochs
NUM_EPOCHS = 50

# Batch size for training and validation
BATCH_SIZE = 4

# Learning rate for the optimizer
LEARNING_RATE = 1e-5

# Optimizer choice (e.g., "AdamW", "SGD")
OPTIMIZER = "AdamW"

# Path to save trained models and logs
OUTPUT_DIR = "../output/" # [1]
MODEL_SAVE_PATH = OUTPUT_DIR + "Mask2Former.ckpt" # [1]

# Image size for resizing (e.g., 512 for smaller datasets, 1024 for Cityscapes)
# Mask2Former typically works with higher resolutions
IMAGE_SIZE = (1024, 1024) # Cityscapes resolution is 1024x2048, so adjust as needed

# --- Mask2Former Model Configuration ---
# Pre-trained Mask2Former model name from Hugging Face
# Example for Cityscapes semantic segmentation:
# "facebook/mask2former-swin-tiny-cityscapes-semantic" [4]
# "facebook/mask2former-swin-base-cityscapes-semantic"
# "facebook/mask2former-swin-large-cityscapes-semantic"
MODEL_NAME = "facebook/mask2former-swin-tiny-cityscapes-semantic"

# Number of classes for your specific dataset
# For Cityscapes semantic segmentation, it's 19 (or 30 if including ignored classes)
# For ADE20K, it's 150 [5]
NUM_CLASSES = 19 # Example for Cityscapes semantic segmentation

# Dimension of Mask2Former's queries/features (e.g., 256 for Swin-Tiny)
PROMPT_EMBEDDING_DIM = 256

# --- Dataset Paths Configuration ---
# IMPORTANT: Update these paths to where you extracted your datasets on the VM
CITYSCAPES_ROOT = "/content/drive/MyDrive/Paper2/Dataset" #
ADE20K_ROOT = "/home/YOUR_USERNAME/Finetune-Mask2Former/data/ade20k" #
COCO_ROOT = "/home/YOUR_USERNAME/Finetune-Mask2Former/data/coco" #

# --- Dataset Specifics (for visualization and processing) ---
# These lists are crucial for dataset loading and visualization [2]
# Example for Cityscapes (semantic segmentation, 19 classes + background/void)
# Adjust based on the specific task (semantic, instance, panoptic) and dataset
ALL_CLASSES_CITYSCAPES = [
    'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail',
    'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
    'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer',
    'train', 'motorcycle', 'bicycle', 'license plate', 'unlabeled', 'ego vehicle'
]
# Map to 19 relevant classes for semantic segmentation, ignoring others or mapping to void
# This mapping depends on how your dataset loader handles classes.
# For Cityscapes, often 19 classes are used, with 'unlabeled' and 'ego vehicle' ignored.
# You might need to define a mapping from raw IDs to training IDs.
# Example: If using 19 classes, you'd map the relevant 19 IDs.
# For simplicity, let's assume NUM_CLASSES is the actual number of classes you train on.
# You'll need to define the actual class names that correspond to your NUM_CLASSES.
# For Cityscapes 19 classes:
CITYSCAPES_19_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

# LABEL_COLORS_LIST: RGB values for each class in the dataset annotations [2]
# This is highly specific to how your dataset's ground truth masks are encoded.
# For Cityscapes, these are standard. You'll need to find the official color map.
# Example (partial, you need the full 19 or 30 class map):
LABEL_COLORS_CITYSCAPES = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
    #... add more for all 19 classes
]

# VIS_LABEL_MAP: Colors for visualization during inference [2]
# Often same as LABEL_COLORS_LIST, but can be adjusted for better contrast.
VIS_LABEL_MAP_CITYSCAPES = LABEL_COLORS_CITYSCAPES # Or a modified version

# --- HMR-PT Specific Configuration ---
# Parameters for Hierarchical Prompt Integration
PROMPT_LENGTH_COARSE = 10  # Number of tokens for coarse-grained prompts
PROMPT_LENGTH_FINE = 5   # Number of tokens for fine-grained prompts

# Number of refinement steps in the Adaptive Mask-Guided Refinement Module
REFINEMENT_STEPS = 3

# VLM model for Semantic-Aware Prompt Initialization (e.g., CLIP)
# Ensure this model is installed and accessible (e.g., `pip install transformers clip`)
SEMANTIC_INIT_VLM_MODEL = "openai/clip-vit-base-patch32"

# Loss weights for HMR-PT's custom loss function
LAMBDA_INITIAL_LOSS = 1.0      # Weight for initial Mask2Former prediction loss
LAMBDA_REFINED_LOSS = 1.0      # Weight for refined mask prediction loss
LAMBDA_BOUNDARY_LOSS = 0.5     # Weight for boundary-specific loss [6]
LAMBDA_CONTRASTIVE_LOSS = 0.1  # Weight for semantic-aware prompt contrastive loss [7, 8, 6]

# --- Two-Stage Training Configuration (Optional) ---
# If you plan to implement a two-stage training strategy
TWO_STAGE_TRAINING = False
STAGE1_EPOCHS = 20 # Epochs for initial prompt learning
STAGE2_EPOCHS = 30 # Epochs for full model fine-tuning with refinement
FREEZE_REFINEMENT_MODULE_STAGE1 = True # Freeze refinement module in stage 1

# --- Data Augmentation (Placeholder, implement in data_transforms.py) ---
# You'll define your Albumentations or torchvision transforms here
# For Mask2Former, resizing, normalization are handled by ImageProcessor [2]
# But you might add other augmentations like random crop, flip, color jitter.
# IMG_TRANSFORMS = {
#     "train":...,
#     "val":...
# }
