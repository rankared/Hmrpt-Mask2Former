# hmr_pt/models/refinement_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicPromptGenerator(nn.Module):
    # --- FIX: It no longer needs 'input_channels' ---
    def __init__(self, embedding_dim):
        super().__init__()
        # --- FIX: The input channels are now hardcoded to 1 ---
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, initial_mask):
        probs = torch.softmax(initial_mask, dim=1).clamp(min=1e-9)
        log_probs = torch.log(probs)
        entropy_map = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        
        # This uncertainty_map correctly has 1 channel
        uncertainty_map = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min())

        x = self.conv1(uncertainty_map)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        dynamic_prompt = self.fc(x)
        return dynamic_prompt.unsqueeze(1)

class IterativeRefinementBlock(nn.Module):
    def __init__(self, input_channels, embedding_dim):
        super().__init__()
        self.conv_refine1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(input_channels)
        self.relu = nn.ReLU()
        self.conv_refine2 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        
        self.prompt_transform = nn.Linear(embedding_dim, input_channels * 2)

    def forward(self, mask_features, dynamic_prompt):
        gamma_beta = self.prompt_transform(dynamic_prompt).squeeze(1)
        gamma = gamma_beta[:, :mask_features.shape[1]].unsqueeze(-1).unsqueeze(-1)
        beta = gamma_beta[:, mask_features.shape[1]:].unsqueeze(-1).unsqueeze(-1)

        x = self.norm1(mask_features)
        x = gamma * x + beta
        
        x = self.relu(self.conv_refine1(x))
        x = self.conv_refine2(x)
        return mask_features + x

class AdaptiveMaskRefinementModule(nn.Module):
    def __init__(self, input_channels, output_channels, embedding_dim, refinement_steps):
        super().__init__()
        # --- FIX: Pass only 'embedding_dim' to the generator ---
        self.dynamic_prompt_generator = DynamicPromptGenerator(embedding_dim)
        self.refinement_steps = refinement_steps
        
        self.initial_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        
        self.refinement_blocks = nn.ModuleList([
            IterativeRefinementBlock(input_channels=input_channels, embedding_dim=embedding_dim)
            for _ in range(refinement_steps)
        ])

        self.output_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, initial_pred_masks, image_features):
        dynamic_prompt = self.dynamic_prompt_generator(initial_pred_masks)
        mask_features = self.initial_conv(initial_pred_masks)

        for i in range(self.refinement_steps):
            mask_features = self.refinement_blocks[i](mask_features, dynamic_prompt)
        
        refined_masks = self.output_conv(mask_features)
        return refined_masks