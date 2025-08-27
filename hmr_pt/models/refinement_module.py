import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicPromptGenerator(nn.Module):
    def __init__(self, input_channels, embedding_dim):
        super().__init__()
        # Takes initial mask (e.g., uncertainty map or error map) and generates prompts
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, embedding_dim) # Generates a single prompt vector for the mask

    def forward(self, initial_mask, image_features=None):
        # initial_mask: (B, C, H, W) - e.g., predicted masks or uncertainty maps
        x = self.conv1(initial_mask)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        dynamic_prompt = self.fc(x) # (B, embedding_dim)
        return dynamic_prompt.unsqueeze(1) # (B, 1, embedding_dim) for concatenation/attention

class IterativeRefinementBlock(nn.Module):
    def __init__(self, input_channels, embedding_dim):
        super().__init__()
        # Simple convolutional block for refinement
        self.conv_refine1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(input_channels)
        self.relu = nn.ReLU()
        self.conv_refine2 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        
        # Mechanism to integrate dynamic prompt (e.g., FiLM layer or simple addition)
        self.prompt_transform = nn.Linear(embedding_dim, input_channels * 2) # For FiLM: scale and bias

    def forward(self, mask_features, dynamic_prompt):
        # mask_features: (B, C, H, W)
        # dynamic_prompt: (B, 1, embedding_dim)

        # Transform prompt for FiLM-like conditioning
        gamma_beta = self.prompt_transform(dynamic_prompt).squeeze(1) # (B, input_channels * 2)
        gamma = gamma_beta[:, :mask_features.shape[1]].unsqueeze(-1).unsqueeze(-1)
        beta = gamma_beta[:, mask_features.shape[1]:].unsqueeze(-1).unsqueeze(-1)

        # Apply FiLM conditioning
        x = self.norm1(mask_features)
        x = gamma * x + beta # Apply scale and bias from prompt
        
        x = self.relu(self.conv_refine1(x))
        x = self.conv_refine2(x)
        return mask_features + x # Residual connection

class AdaptiveMaskRefinementModule(nn.Module):
    def __init__(self, input_channels, output_channels, embedding_dim, refinement_steps):
        super().__init__()
        self.dynamic_prompt_generator = DynamicPromptGenerator(input_channels, embedding_dim)
        self.refinement_steps = refinement_steps
        
        self.initial_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1) # Project initial mask
        
        self.refinement_blocks = nn.ModuleList()

        for _ in range(refinement_steps):
            self.refinement_blocks.append(
                IterativeRefinementBlock(
                    input_channels=input_channels, 
                    embedding_dim=embedding_dim
                )
            )

        self.output_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, initial_pred_masks, image_features):
        # initial_pred_masks: (B, num_queries, H, W) - raw mask predictions from Mask2Former
        # image_features: (B, C_img, H_img, W_img) - image features from backbone (e.g., pixel decoder output)

        # Convert initial_pred_masks to a single mask representation for prompt generation
        # For panoptic/instance, this might involve taking the argmax or top-k masks
        # For semantic, it's usually (B, num_classes, H, W)
        # Let's assume initial_pred_masks are already (B, num_classes, H, W) for simplicity
        # If it's (B, num_queries, H, W), we might need to collapse it or process per-query.
        
        # For simplicity, let's assume initial_pred_masks are already semantic-like (B, C, H, W)
        # Or we can take the most confident mask for instance-level refinement.
        
        # Generate dynamic prompt based on initial mask quality (e.g., uncertainty or error regions)
        # This could involve computing uncertainty from initial_pred_masks
        # For now, we'll just pass the initial_pred_masks directly to the prompt generator
        dynamic_prompt = self.dynamic_prompt_generator(initial_pred_masks) # (B, 1, embedding_dim)

        # Initial mask features for refinement
        mask_features = self.initial_conv(initial_pred_masks)

        # Iterative refinement
        for i in range(self.refinement_steps):
            mask_features = self.refinement_blocks[i](mask_features, dynamic_prompt)
        
        refined_masks = self.output_conv(mask_features)
        return refined_masks
