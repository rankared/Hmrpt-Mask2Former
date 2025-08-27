import torch
import torch.nn as nn
import torch.nn.functional as F

class HMRLoss(nn.Module):
    def __init__(self, num_classes, lambda_initial=1.0, lambda_refined=1.0, lambda_boundary=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_initial = lambda_initial
        self.lambda_refined = lambda_refined
        self.lambda_boundary = lambda_boundary

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.dice_loss = self._dice_loss

    def _dice_loss(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)
        
        # DEFINITIVE FIX: Replace .view() with .reshape() to handle non-contiguous tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

    def forward(self, initial_pred_masks, refined_pred_masks, initial_class_logits, labels):
        target_masks_resized = F.interpolate(labels.unsqueeze(1).float(), size=initial_pred_masks.shape[-2:], mode='nearest').squeeze(1).long()
        
        valid_pixel_mask = (target_masks_resized != 255)
        target_for_one_hot = target_masks_resized.clone()
        target_for_one_hot[~valid_pixel_mask] = 0 
        
        target_masks_one_hot = F.one_hot(target_for_one_hot, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        target_masks_one_hot = target_masks_one_hot * valid_pixel_mask.unsqueeze(1)

        loss_initial_mask = self.ce_loss(initial_pred_masks, target_masks_resized) + self.dice_loss(initial_pred_masks, target_masks_one_hot)
        loss_refined_mask = self.ce_loss(refined_pred_masks, target_masks_resized) + self.dice_loss(refined_pred_masks, target_masks_one_hot)

        total_loss = (self.lambda_initial * loss_initial_mask +
                      self.lambda_refined * loss_refined_mask)
          
        return total_loss