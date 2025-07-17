import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer # For VLM-guided initialization [5, 6]

class HierarchicalPromptLayer(nn.Module):
    def __init__(self, prompt_length, prompt_embedding_dim, is_coarse=True):
        super().__init__()
        self.prompt_length = prompt_length
        self.prompt_embedding_dim = prompt_embedding_dim
        self.is_coarse = is_coarse

        # Learnable prompt tokens
        self.prompts = nn.Parameter(torch.randn(1, prompt_length, prompt_embedding_dim))

        # Simple MLP to integrate prompts with input features (e.g., query features)
        self.integration_mlp = nn.Sequential(
            nn.Linear(prompt_embedding_dim * 2, prompt_embedding_dim),
            nn.ReLU(),
            nn.Linear(prompt_embedding_dim, prompt_embedding_dim)
        )

    def forward(self, input_features, intermediate_mask=None):
        # input_features: (batch_size, num_queries, embedding_dim)
        batch_size = input_features.shape
        
        # Expand prompts to batch size
        prompts_batch = self.prompts.expand(batch_size, -1, -1)

        # Concatenate prompts with input features (e.g., object queries)
        # This is a simplified concatenation. In a real Transformer, prompts are usually prepended
        # or interact via cross-attention. Here, we'll integrate them.
        
        # For fine-grained prompts, condition on intermediate_mask (e.g., uncertainty or error regions)
        if not self.is_coarse and intermediate_mask is not None:
            # Example: Simple spatial pooling of mask to get a mask-guided feature
            # This would be a more complex module in reality, e.g., a small CNN or attention over mask.
            mask_feature = torch.mean(intermediate_mask, dim=(-1, -2)).unsqueeze(1) # (B, 1, C)
            mask_feature = mask_feature.expand(-1, prompts_batch.shape[1], -1) # (B, prompt_length, C)
            
            # Combine prompts with mask-guided features
            combined_prompts = torch.cat([prompts_batch, mask_feature], dim=-1)
            processed_prompts = self.integration_mlp(combined_prompts)
        else:
            processed_prompts = prompts_batch # For coarse prompts or if no mask is provided

        # Integrate processed prompts with input features (e.g., by adding or concatenating)
        # This is a conceptual integration. In Mask2Former, prompts would typically modify
        # the query features before they enter the Transformer decoder layers.
        # For simplicity, let's assume we add them to the input features.
        # A more sophisticated approach might involve cross-attention between prompts and queries.
        
        # Example: Simple addition (conceptual)
        # This requires `processed_prompts` to be broadcastable or reshaped to match `input_features`
        # For Mask2Former, prompts would typically be prepended to the query sequence or interact via cross-attention.
        # Here, we'll assume they modify the query features directly.
        
        # A more realistic integration for Mask2Former would be to modify the query features
        # that are passed through the decoder layers.
        # For example, if input_features are the object queries:
        # new_input_features = input_features + processed_prompts[:, :input_features.shape[1], :]
        # This requires careful dimension matching.

        # For now, let's return the processed prompts to be used in the Mask2Former decoder's query path
        return processed_prompts # These prompts will be injected into Mask2Former's decoder

class SemanticPromptInitializer(nn.Module):
    def __init__(self, num_classes, prompt_embedding_dim, vlm_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.num_classes = num_classes
        self.prompt_embedding_dim = prompt_embedding_dim

        # Load pre-trained VLM for text embeddings [5, 6]
        self.tokenizer = CLIPTokenizer.from_pretrained(vlm_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(vlm_model_name)
        # Freeze VLM text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Learnable projection to map VLM embeddings to prompt_embedding_dim
        self.projection = nn.Linear(self.text_encoder.config.hidden_size, prompt_embedding_dim)

    def get_semantic_prompts(self, class_names):
        # Generate text embeddings for class names
        inputs = self.tokenizer(class_names, padding=True, return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.text_encoder(**inputs).last_hidden_state[:, 0, :] # token embedding
        
        # Project to desired prompt dimension
        semantic_prompts = self.projection(text_embeddings) # (num_classes, prompt_embedding_dim)
        return semantic_prompts

    def forward(self, class_names):
        # This forward is conceptual for initialization.
        # In practice, this would be called once to initialize the `self.prompts` in HierarchicalPromptLayer
        return self.get_semantic_prompts(class_names)
