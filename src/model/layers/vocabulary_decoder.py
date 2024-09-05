import torch
import torch.nn as nn
import torch.nn.functional as F

class VocabularyDecoder(nn.Module):
    """A transformer-based multimodal decoder that selects the most similar trajectory from a vocabulary"""

    def __init__(self, embed_dim, future_steps) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.future_steps = future_steps

        # Linear projection for query, key, and value
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # MLP for the final output
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, future_steps * 2)
        )
        
        self.confidence_layer = nn.Linear(embed_dim, 1)  # 각 trajectory에 대한 confidence score 출력
        

    def forward(self, query, key, value):
        # Project query, key, value
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        print("query")
        print(query.shape)
        print("key")
        print(key.shape)
        print(key)
        print("value")
        print(value.shape)
        print(value)

        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        print("attention_scores")
        print(attention_scores)
        attention_weights = F.softmax(attention_scores)

        print("attention_weights")
        print(attention_weights)
        
        # Weighted sum of the values
        weighted_sum = torch.matmul(attention_weights, value)
        print("weighted_sum")
        print(weighted_sum)
        # Pass the weighted sum through the output MLP
        # output = self.output_layer(weighted_sum)
        confidence_scores = self.confidence_layer(weighted_sum).squeeze(-1)  # (B, vocabulary_size)
        print("confidence_scores before softmax")
        print(confidence_scores)
        confidence_scores = F.softmax(confidence_scores, dim=-1)
        print("confidence_scores after softmax")
        print(confidence_scores)
        
        return confidence_scores