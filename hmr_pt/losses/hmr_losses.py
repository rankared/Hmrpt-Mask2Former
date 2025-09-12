# hmr_pt/losses/hmr_losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class HMRLoss(nn.Module):
    def __init__(self, num_classes, lambda_initial=1.0, lambda_refined=1.0, lambda_boundary=0.5, lambda_contrastive=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_initial = lambda_initial
        self.lambda_refined = lambda_refined
        self.lambda_boundary = lambda_boundary
        self.lambda_contrastive = lambda_contrastive

        # [TRACE] Print initialized loss weights for verification
        print(f"[TRACE] HMRLoss initialized with weights: initial={lambda_initial}, refined={lambda_refined}, boundary={lambda_boundary}, contrastive={lambda_contrastive}")

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.dice_loss = self._dice_loss

        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_kernel_x', sobel_kernel_x)
        self.register_buffer('sobel_kernel_y', sobel_kernel_y)

    def _dice_loss(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

    def _boundary_loss(self, refined_pred_masks, target_masks_one_hot, boundary_targets):
        pred_boundary = refined_pred_masks * boundary_targets
        target_boundary = target_masks_one_hot * boundary_targets
        return self.dice_loss(pred_boundary, target_boundary)


    def _contrastive_loss(self, class_logits, gt_labels):
        """
        Calculates a contrastive loss to make class query embeddings more discriminative.
        This is done by creating a proxy N-way classification task, where N is the
        number of unique classes present in the ground truth.
        """
        loss = 0.0
        num_items = 0
        
        for i in range(class_logits.shape[0]): # Iterate over batch
            present_labels = torch.unique(gt_labels[i])
            present_labels = present_labels[present_labels != 255] # Ignore the ignore_index
            
            if len(present_labels) < 2:
                continue

            k = len(present_labels) # Number of present classes
            
            # Get the mean logit across all queries for each of the k present classes
            # This gives a representative logit vector for the image.
            query_logits_for_item = class_logits[i, :, present_labels].mean(dim=0) # Shape: (k)
            
            # The target is a set of unique indices for our k classes
            target = torch.arange(k, device=class_logits.device) # Shape: (k), Values: [0, 1, ..., k-1]

            # 
            # 
            # 
            # To create a k-way classification task, we repeat the logit vector
            # k times, creating an input of shape (k, k). The cross_entropy function
            # will then compare the i-th repeated logit vector to the target label i.
            # This encourages the logit for class i to be highest in the i-th row.
            input_logits = query_logits_for_item.unsqueeze(0).repeat(k, 1) # Shape: (k, k)

            # This call is now valid: input is (N, C) -> (k, k) and target is (N) -> (k)
            loss += F.cross_entropy(input_logits, target)
            num_items += 1

        return loss / (num_items + 1e-6)


    def forward(self, initial_pred_masks, refined_pred_masks, initial_class_logits, labels):
        target_masks_resized = F.interpolate(labels.unsqueeze(1).float(), size=initial_pred_masks.shape[-2:], mode='nearest').squeeze(1).long()

        with torch.no_grad():
            boundary_targets_x = F.conv2d(target_masks_resized.float().unsqueeze(1), self.sobel_kernel_x, padding=1)
            boundary_targets_y = F.conv2d(target_masks_resized.float().unsqueeze(1), self.sobel_kernel_y, padding=1)
            boundary_targets = ((boundary_targets_x.abs() > 0) | (boundary_targets_y.abs() > 0)).float()

        valid_pixel_mask = (target_masks_resized != 255)
        target_for_one_hot = target_masks_resized.clone()
        target_for_one_hot[~valid_pixel_mask] = 0
        target_masks_one_hot = F.one_hot(target_for_one_hot, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        target_masks_one_hot = target_masks_one_hot * valid_pixel_mask.unsqueeze(1)

        loss_initial_mask = self.ce_loss(initial_pred_masks, target_masks_resized) + self.dice_loss(initial_pred_masks, target_masks_one_hot)
        loss_refined_mask = self.ce_loss(refined_pred_masks, target_masks_resized) + self.dice_loss(refined_pred_masks, target_masks_one_hot)
        loss_boundary = self._boundary_loss(refined_pred_masks, target_masks_one_hot, boundary_targets)
        loss_contrastive = self._contrastive_loss(initial_class_logits, labels)

        ## [TRACE] Print individual loss values before weighting and summing
        #print(
        #    f"\n[TRACE] Loss Components: \n"
        #    f"  - Initial Mask Loss: {loss_initial_mask.item():.4f}\n"
        #    f"  - Refined Mask Loss: {loss_refined_mask.item():.4f}\n"
        #    f"  - Boundary Loss:     {loss_boundary.item():.4f}\n"
        #    f"  - Contrastive Loss:  {loss_contrastive.item():.4f}"
        #)

        total_loss = (self.lambda_initial * loss_initial_mask +
                      self.lambda_refined * loss_refined_mask +
                      self.lambda_boundary * loss_boundary +
                      self.lambda_contrastive * loss_contrastive)
        
        ## [TRACE] Print the final combined loss
        #print(f"[TRACE] Total Loss: {total_loss.item():.4f}\n")
          
        return total_loss