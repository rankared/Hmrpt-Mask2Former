# hmr_pt/models/hmr_mask2former.py

import torch
import torch.nn as nn
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
import copy
from functools import partial

from hmr_pt.models.prompt_modules import HierarchicalPromptLayer, SemanticPromptInitializer
from hmr_pt.models.refinement_module import AdaptiveMaskRefinementModule
from hmr_pt.losses.hmr_losses import HMRLoss
from config import CITYSCAPES_19_CLASSES

class HMRMask2Former(nn.Module):
    def __init__(self, model_name="facebook/mask2former-swin-tiny-cityscapes-semantic",
                 num_classes=19,
                 prompt_length_coarse=10,
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
        
        self.hierarchical_prompts = nn.ModuleList([
            HierarchicalPromptLayer(prompt_length_coarse, prompt_embedding_dim)
            for _ in range(self.num_decoder_layers)
        ])

        self.new_class_predictor = nn.Linear(self.prompt_embedding_dim, self.num_classes + 1)
        self.new_mask_embedder = copy.deepcopy(self.mask2former.model.transformer_module.decoder.mask_predictor.mask_embedder)

        self.refinement_module = AdaptiveMaskRefinementModule(
            input_channels=self.num_classes,
            output_channels=self.num_classes,
            embedding_dim=prompt_embedding_dim,
            refinement_steps=refinement_steps
        )

        self.semantic_initializer = SemanticPromptInitializer(
            num_classes=len(CITYSCAPES_19_CLASSES),
            prompt_embedding_dim=prompt_embedding_dim,
            vlm_model_name=semantic_init_vlm_model
        )
        self.initialize_prompts_semantically(CITYSCAPES_19_CLASSES)
        
        self.criterion = HMRLoss(
            num_classes=num_classes,
            lambda_initial=lambda_initial,
            lambda_refined=lambda_refined,
            lambda_boundary=lambda_boundary,
            lambda_contrastive=lambda_contrastive
        )
        
        # This is the new, robust method to inject prompts
        self._inject_prompts_into_decoder()

    def initialize_prompts_semantically(self, class_names):
        semantic_prompts = self.semantic_initializer(class_names)
        with torch.no_grad():
            for prompt_layer in self.hierarchical_prompts:
                num_prompts_in_layer = prompt_layer.prompts.shape[1]
                indices = torch.randint(0, len(class_names), (num_prompts_in_layer,))
                selected_prompts = semantic_prompts[indices]
                prompt_layer.prompts.data = selected_prompts.unsqueeze(0)

    def _create_prompted_forward(self, original_forward, prompt_layer):
        # This is a wrapper function. It takes the original forward method of a decoder layer
        # and returns a NEW forward method that first applies our prompts.
        def prompted_forward(*args, **kwargs):
            # The queries are the first argument ('hidden_states')
            queries = args[0]
            
            # Apply our learnable prompt
            prompted_queries = prompt_layer(queries)
            
            # Replace the original queries with our prompted ones
            new_args = (prompted_queries,) + args[1:]
            
            # Call the original decoder layer's forward method with the prompted queries
            return original_forward(*new_args, **kwargs)
        
        return prompted_forward

    def _inject_prompts_into_decoder(self):
        # Iterate through each of the 9 decoder layers
        decoder_layers = self.mask2former.model.transformer_module.decoder.layers
        for i, layer in enumerate(decoder_layers):
            # Get the original forward method of this layer
            original_forward = layer.forward
            
            # Create a new, prompted forward method using our wrapper
            prompted_forward_func = self._create_prompted_forward(original_forward, self.hierarchical_prompts[i])
            
            # Replace the layer's original forward method with our new one
            layer.forward = prompted_forward_func

    def forward(self, pixel_values, pixel_mask=None, labels=None, task_type="semantic"):
        # With the prompts injected, we can now call the model in a much simpler way
        outputs = self.mask2former(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_hidden_states=True, # We still need the hidden states
            return_dict=True
        )
        
        # The decoder has now run with our prompts. We can get the final queries.
        final_queries = outputs.transformer_decoder_hidden_states[-1]
        final_queries_permuted = final_queries.permute(1, 0, 2)
        
        mask_embed = outputs.pixel_decoder_last_hidden_state
        
        # Use the prompted final queries to get predictions
        initial_class_logits = self.new_class_predictor(final_queries_permuted)
        mask_embeddings = self.new_mask_embedder(final_queries_permuted)
        initial_pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embeddings, mask_embed)

        class_logits = initial_class_logits[:, :, :self.num_classes]
        mask_logits_reshaped = initial_pred_masks.flatten(2)
        mask_probs_reshaped = torch.softmax(mask_logits_reshaped, dim=1)
        semantic_initial_masks_flat = torch.einsum("bqc,bqw->bcw", class_logits, mask_probs_reshaped)
        height, width = initial_pred_masks.shape[-2:]
        
        semantic_initial_masks = semantic_initial_masks_flat.view(-1, self.num_classes, height, width)

        refined_pred_masks = self.refinement_module(semantic_initial_masks, mask_embed)

        loss = None
        if labels is not None:
            loss = self.criterion(
                initial_pred_masks=semantic_initial_masks,
                refined_pred_masks=refined_pred_masks,
                initial_class_logits=initial_class_logits,
                labels=labels,
            )

        return {
            "loss": loss,
            "initial_pred_masks": semantic_initial_masks,
            "refined_pred_masks": refined_pred_masks,
            "class_logits": class_logits
        }