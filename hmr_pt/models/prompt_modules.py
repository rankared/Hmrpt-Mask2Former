import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

class HierarchicalPromptLayer(nn.Module):
    def __init__(self, prompt_length, prompt_embedding_dim, is_coarse=True):
        super().__init__()
        self.prompt_length = prompt_length
        self.prompt_embedding_dim = prompt_embedding_dim
        self.is_coarse = is_coarse

        self.prompts = nn.Parameter(torch.randn(1, prompt_length, prompt_embedding_dim))

    def forward(self, input_features, intermediate_mask=None):
        # input_features shape: (num_queries, batch_size, embedding_dim)
        
        modified_features = input_features.clone()
        # Permute prompts to (prompt_length, 1, dim) for broadcasting
        prompts_to_add = self.prompts.permute(1, 0, 2)
        modified_features[:self.prompt_length, :, :] += prompts_to_add
        
        return modified_features

class SemanticPromptInitializer(nn.Module):
    def __init__(self, num_classes, prompt_embedding_dim, vlm_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.num_classes = num_classes
        self.prompt_embedding_dim = prompt_embedding_dim

        self.tokenizer = CLIPTokenizer.from_pretrained(vlm_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(vlm_model_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(self.text_encoder.config.hidden_size, prompt_embedding_dim)

    def get_semantic_prompts(self, class_names):
        device = self.projection.weight.device
        inputs = self.tokenizer(class_names, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            text_embeddings = self.text_encoder(**inputs).last_hidden_state[:, 0, :]
        
        semantic_prompts = self.projection(text_embeddings)
        return semantic_prompts

    def forward(self, class_names):
        return self.get_semantic_prompts(class_names)