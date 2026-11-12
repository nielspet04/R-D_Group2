import torch
from torch import Tensor, nn

class MaskedGRUSTA(nn.Module):
    """predict STA verbs and nouns from chronological video features"""

    def __init__(self, feature_size: int = 512, hidden_size: int = 512, num_verbs: int = 97, num_nouns: int = 300, dropout: float = 0.3,
    )->None:
        super().__init__()
        
        for name, value in (
            ("feature_size", feature_size),
            ("hidden_size", hidden_size),
            ("num_verbs", num_verbs),
            ("num_nouns", num_nouns),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.gru_cell = nn.GRUCell(feature_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.verb_head = nn.Linear(hidden_size, num_verbs)
        self.noun_head = nn.Linear(hidden_size, num_nouns)

    def final_hidden(self, features: Tensor, valid_mask: Tensor
    )-> Tensor: 
        """return the final hidden state, skipping invalid segments"""
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, segments, feature_size]")

        if features.shape[-1] != self.feature_size:
            raise ValueError(
                f"expected feature size {self.feature_size}, "
                f"received {features.shape[-1]}"
            )
        if valid_mask.shape != features.shape[:2]:
            raise ValueError(
                "valid_mask must match the batch and segment dimensions"
            )
        
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have dtype torch.bool")

        if not features.is_floating_point():
            raise TypeError("features must be a floating-point tensor")

        if valid_mask.device != features.device:
            raise ValueError("valid_mask and features must be on the same device")

        hidden = features.new_zeros(
            features.shape[0], self.hidden_size
        )

        for step in range(features.shape[1]):
            candidate = self.gru_cell(features[:, step], hidden)
            step_mask = valid_mask[:, step].unsqueeze(-1)
            hidden = torch.where(step_mask, candidate, hidden)

        return hidden

    def forward(self, features: Tensor, valid_mask: Tensor,
    )-> tuple[Tensor, Tensor]:
        hidden = self.final_hidden(features, valid_mask)
        representation = self.dropout(hidden)

        verb_logits = self.verb_head(representation)
        noun_logits = self.noun_head(representation)
        return verb_logits, noun_logits
    
