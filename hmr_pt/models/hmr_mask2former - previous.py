# hmr_pt/models/hmr_mask2former.py

import torch
import torch.nn as nn
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
from functools import partial
import copy

from hmr_pt.models.prompt_modules import HierarchicalPromptLayer, SemanticPromptInitializer
from hmr_pt.models.refinement_module import AdaptiveMaskRefinementModule
from hmr_pt.losses.hmr_losses import HMRLoss

class HMRMask2Former(nn.Module):
    def __init__(self, model_name="facebook/mask2former-swin-tiny-cityscapes-semantic",
                 num_classes=19,
                 prompt_length_coarse=10,
                 prompt_length_fine=5,
                 num_decoder_layers=6,
                 refinement_steps=3,
                 prompt_embedding_dim=256,
                 semantic_init_vlm_model="openai/clip-vit-base-patch32",
                 lambda_initial=1.0,
                 lambda_refined=1.0,
                 lambda_boundary=0.5,
                 lambda_contrastive=0.1):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)

        for param in self.mask2former.parameters():
            param.requires_grad = False
        
        self.num_classes = num_classes
        self.prompt_embedding_dim = prompt_embedding_dim

        self.num_decoder_layers = 9
        self.hierarchical_prompts = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.hierarchical_prompts.append(
                HierarchicalPromptLayer(prompt_length_coarse, prompt_embedding_dim)
            )

        self.new_class_predictor = nn.Linear(self.prompt_embedding_dim, self.num_classes + 1)
        self.new_mask_embedder = copy.deepcopy(self.mask2former.model.transformer_module.decoder.mask_predictor.mask_embedder)

        self.refinement_module = AdaptiveMaskRefinementModule(
            input_channels=self.num_classes,
            output_channels=self.num_classes,
            embedding_dim=prompt_embedding_dim,
            refinement_steps=refinement_steps
        )
        
        self.criterion = HMRLoss(
            num_classes=num_classes,
            lambda_initial=lambda_initial,
            lambda_refined=lambda_refined,
            lambda_boundary=lambda_boundary,
            lambda_contrastive=lambda_contrastive
        )

# In the HMRMask2Former class, replace the entire 'forward' method with this one:
    def forward(self, pixel_values, pixel_mask=None, labels=None, task_type="semantic"):
        # 1. Get base model outputs
        outputs = self.mask2former(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        decoder_hidden_states = outputs.transformer_decoder_hidden_states
        mask_embed = outputs.pixel_decoder_last_hidden_state

        # 2. Apply hierarchical prompts to create 'modified_queries'
        modified_queries = decoder_hidden_states[0]
        for i, (layer_output, prompt_layer) in enumerate(zip(decoder_hidden_states, self.hierarchical_prompts)):
             modified_queries = prompt_layer(layer_output)

        # 3. Permute queries for downstream modules
        modified_queries = modified_queries.permute(1, 0, 2)
        
        # 4. Use 'modified_queries' to get class and mask predictions
        initial_class_logits = self.new_class_predictor(modified_queries)
        mask_embeddings = self.new_mask_embedder(modified_queries)
        initial_pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embeddings, mask_embed)

        class_logits = initial_class_logits[:, :, :self.num_classes]
        
        mask_logits_reshaped = initial_pred_masks.flatten(2)

        # 5. Apply softmax to normalize mask predictions and prevent exploding loss
        mask_probs_reshaped = torch.softmax(mask_logits_reshaped, dim=1)
        
        # 6. Create the initial semantic mask using normalized probabilities
        semantic_initial_masks_flat = torch.einsum("bqc,bqw->bcw", class_logits, mask_probs_reshaped)

        height, width = initial_pred_masks.shape[-2:]
        semantic_initial_masks = semantic_initial_masks_flat.view(-1, self.num_classes, height, width)

        # 7. Refine the mask
        refined_pred_masks = self.refinement_module(semantic_initial_masks, pixel_values)

        # 8. Calculate loss if labels are provided
        loss = None
        if labels is not None:
            ## [TRACE] Print the shapes of all tensors being passed to the loss function
            #print(
            #    f"\n[TRACE] Shapes before loss calculation:\n"
            #    f"  - Initial Pred Masks: {semantic_initial_masks.shape}\n"
            #    f"  - Refined Pred Masks: {refined_pred_masks.shape}\n"
            #    f"  - Initial Class Logits: {initial_class_logits.shape}\n"
            #    f"  - Labels: {labels.shape}"
            #)

            loss = self.criterion(
                initial_pred_masks=semantic_initial_masks,
                refined_pred_masks=refined_pred_masks,
                initial_class_logits=initial_class_logits,
                labels=labels,
            )

        return {
            "loss": loss,
            "initial_pred_masks": initial_pred_masks,
            "refined_pred_masks": refined_pred_masks,
            "class_logits": class_logits
        }
