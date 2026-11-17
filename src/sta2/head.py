"""Trainable attentive prediction head for STA2."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class MLP(nn.Module):
    """Two-layer MLP used inside the attentive pooler."""

    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = self.fc1(inputs)
        inputs = self.activation(inputs)
        inputs = self.dropout(inputs)
        inputs = self.fc2(inputs)
        return self.dropout(inputs)


class SelfAttention(nn.Module):
    """Multi-head self-attention using PyTorch SDPA."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_dropout = attention_dropout

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, token_count, embed_dim = inputs.shape

        qkv = self.qkv(inputs)
        qkv = qkv.reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)

        queries, keys, values = qkv.unbind(dim=0)

        dropout_probability = (
            self.attention_dropout if self.training else 0.0
        )

        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            dropout_p=dropout_probability,
        )

        attended = attended.transpose(1, 2).reshape(
            batch_size,
            token_count,
            embed_dim,
        )
        attended = self.proj(attended)
        return self.projection_dropout(attended)


class SelfAttentionBlock(nn.Module):
    """Pre-normalized self-attention and MLP residual block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs + self.attn(self.norm1(inputs))
        inputs = inputs + self.mlp(self.norm2(inputs))
        return inputs


class CrossAttention(nn.Module):
    """Cross-attention from learned queries to video tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_projection = nn.Linear(
            embed_dim,
            embed_dim,
            bias=qkv_bias,
        )
        self.key_value_projection = nn.Linear(
            embed_dim,
            2 * embed_dim,
            bias=qkv_bias,
        )

    def forward(
        self,
        queries: Tensor,
        context: Tensor,
    ) -> Tensor:
        batch_size, query_count, embed_dim = queries.shape
        context_count = context.shape[1]

        queries = self.query_projection(queries)
        queries = queries.reshape(
            batch_size,
            query_count,
            self.num_heads,
            self.head_dim,
        )
        queries = queries.permute(0, 2, 1, 3)

        key_values = self.key_value_projection(context)
        key_values = key_values.reshape(
            batch_size,
            context_count,
            2,
            self.num_heads,
            self.head_dim,
        )
        key_values = key_values.permute(2, 0, 3, 1, 4)

        keys, values = key_values.unbind(dim=0)

        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
        )

        return attended.transpose(1, 2).reshape(
            batch_size,
            query_count,
            embed_dim,
        )


class CrossAttentionBlock(nn.Module):
    """Learned-query cross-attention followed by an MLP."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()

        self.context_norm = nn.LayerNorm(embed_dim)
        self.cross_attention = CrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.query_norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
        )

    def forward(
        self,
        queries: Tensor,
        context: Tensor,
    ) -> Tensor:
        queries = queries + self.cross_attention(
            queries,
            self.context_norm(context),
        )
        queries = queries + self.mlp(self.query_norm(queries))
        return queries


class AttentivePooler(nn.Module):
    """V-JEPA-style attentive pooler with learned output queries."""

    def __init__(
        self,
        embed_dim: int,
        num_queries: int = 3,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        depth: int = 4,
        use_activation_checkpointing: bool = True,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()

        self.use_activation_checkpointing = (
            use_activation_checkpointing
        )

        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_queries, embed_dim)
        )

        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth - 1)
            ]
        )

        self.cross_attention_block = CrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        self.init_std = init_std
        self.apply(self._initialize_weights)
        nn.init.trunc_normal_(self.query_tokens, std=init_std)
        self._rescale_blocks()

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _rescale_blocks(self) -> None:
        """Apply the residual-branch scaling used upstream."""

        for layer_index, block in enumerate(self.blocks, start=1):
            scale = math.sqrt(2.0 * layer_index)
            block.attn.proj.weight.data.div_(scale)
            block.mlp.fc2.weight.data.div_(scale)

        cross_layer_index = max(len(self.blocks), 1)
        cross_scale = math.sqrt(2.0 * cross_layer_index)
        self.cross_attention_block.mlp.fc2.weight.data.div_(
            cross_scale
        )

    def forward(self, tokens: Tensor) -> Tensor:
        for block in self.blocks:
            if self.use_activation_checkpointing and self.training:
                tokens = checkpoint(
                    block,
                    tokens,
                    use_reentrant=False,
                )
            else:
                tokens = block(tokens)

        queries = self.query_tokens.expand(
            tokens.shape[0],
            -1,
            -1,
        )

        return self.cross_attention_block(queries, tokens)


class AnticipationHead(nn.Module):
    """Predict verbs, nouns and auxiliary action pairs from video tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_actions: int,
        num_verbs: int = 97,
        num_nouns: int = 300,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        depth: int = 4,
        use_activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()

        dimensions = {
            "embed_dim": embed_dim,
            "num_actions": num_actions,
            "num_verbs": num_verbs,
            "num_nouns": num_nouns,
            "num_heads": num_heads,
            "depth": depth,
        }

        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads"
            )

        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")

        self.embed_dim = embed_dim
        self.num_actions = num_actions

        self.pooler = AttentivePooler(
            embed_dim=embed_dim,
            num_queries=3,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            use_activation_checkpointing=(
                use_activation_checkpointing
            ),
        )

        self.verb_classifier = nn.Linear(embed_dim, num_verbs)
        self.noun_classifier = nn.Linear(embed_dim, num_nouns)
        self.action_classifier = nn.Linear(embed_dim, num_actions)

    def _validate_tokens(
        self,
        observed_tokens: Tensor,
        future_tokens: Tensor,
    ) -> None:
        for name, tokens in (
            ("observed_tokens", observed_tokens),
            ("future_tokens", future_tokens),
        ):
            if tokens.ndim != 3:
                raise ValueError(
                    f"{name} must have shape [batch, tokens, embed_dim]"
                )

            if tokens.shape[-1] != self.embed_dim:
                raise ValueError(
                    f"{name} has embedding dimension "
                    f"{tokens.shape[-1]}, expected {self.embed_dim}"
                )

            if not tokens.is_floating_point():
                raise TypeError(f"{name} must be floating point")

        if observed_tokens.shape[0] != future_tokens.shape[0]:
            raise ValueError(
                "observed_tokens and future_tokens must have "
                "the same batch size"
            )

        if observed_tokens.device != future_tokens.device:
            raise ValueError(
                "observed_tokens and future_tokens must be "
                "on the same device"
            )

        if observed_tokens.dtype != future_tokens.dtype:
            raise ValueError(
                "observed_tokens and future_tokens must have "
                "the same dtype"
            )

        total_tokens = (
            observed_tokens.shape[1] + future_tokens.shape[1]
        )
        if total_tokens == 0:
            raise ValueError(
                "at least one observed or future token is required"
            )

    def forward(
        self,
        observed_tokens: Tensor,
        future_tokens: Tensor,
    ) -> dict[str, Tensor]:
        self._validate_tokens(observed_tokens, future_tokens)

        tokens = torch.cat(
            (observed_tokens, future_tokens),
            dim=1,
        )

        pooled_queries = self.pooler(tokens)

        verb_representation = pooled_queries[:, 0]
        noun_representation = pooled_queries[:, 1]
        action_representation = pooled_queries[:, 2]

        return {
            "verb_logits": self.verb_classifier(
                verb_representation
            ),
            "noun_logits": self.noun_classifier(
                noun_representation
            ),
            "action_logits": self.action_classifier(
                action_representation
            ),
        }