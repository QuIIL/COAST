from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _COASTBase(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        feat_dim: int,
        num_genes: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        add_type_embed: bool,
        type_embedding: str,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.add_type_embed = add_type_embed
        self.type_embedding = type_embedding

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(dropout),
        )

        if add_type_embed and type_embedding == "affine":
            self.type_gamma = nn.Embedding(2, hidden_dim)
            self.type_beta = nn.Embedding(2, hidden_dim)
        elif add_type_embed and type_embedding == "additive":
            self.type_beta = nn.Embedding(2, hidden_dim)

        self.q_norm_layers = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        )
        self.kv_norm_layers = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        )
        self.attn_layers = nn.ModuleList(
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        )
        self.ffn_layers = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        )

        self.proj_norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, feat_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(dropout),
        )
        self.pred_head = nn.Linear(feat_dim, num_genes)
        self.diff_head = nn.Linear(feat_dim, num_genes)

    def _add_type_embedding(
        self,
        context: torch.Tensor,
        context_type: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.add_type_embed or self.type_embedding == "none":
            return context
        if context_type is None:
            raise ValueError("ctx_type is required when add_type_embed=True")

        beta = self.type_beta(context_type.long())
        if self.type_embedding == "additive":
            return context + beta

        gamma = self.type_gamma(context_type.long())
        return context * (1 + gamma) + beta

    def forward(
        self,
        x_img: torch.Tensor,
        x_ctx_img: torch.Tensor,
        coords: torch.Tensor,
        coords_ctx: torch.Tensor,
        ctx_type: Optional[torch.Tensor] = None,
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, context_size, input_dim = x_ctx_img.shape

        query = self.head(x_img).unsqueeze(1)
        context = self.head(x_ctx_img.reshape(batch_size * context_size, input_dim))
        context = context.reshape(batch_size, context_size, -1)
        context = self._add_type_embedding(context, ctx_type)
        sequence = torch.cat([query, context], dim=1)

        for layer_index in range(self.num_layers):
            residual = sequence
            normalized = self.q_norm_layers[layer_index](sequence)
            attention, _ = self.attn_layers[layer_index](
                query=normalized,
                key=normalized,
                value=normalized,
                key_padding_mask=None,
                need_weights=False,
            )
            sequence = residual + attention
            sequence = sequence + self.ffn_layers[layer_index](sequence)

        query = sequence[:, 0, :]
        context = sequence[:, 1:, :]
        spot_features = self.proj(self.proj_norm(query))
        context_features = self.proj(self.proj_norm(context))
        difference_features = spot_features.unsqueeze(1) - context_features

        spot_prediction = self.pred_head(spot_features)
        difference_prediction = self.diff_head(difference_features)
        return spot_prediction, difference_prediction


class COAST(_COASTBase):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        feat_dim: int = 1024,
        num_genes: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        add_type_embed: bool = True,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            feat_dim=feat_dim,
            num_genes=num_genes,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            add_type_embed=add_type_embed,
            type_embedding="affine",
        )


class COASTEmbAbl(_COASTBase):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        feat_dim: int = 1024,
        num_genes: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        add_type_embed: bool = True,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            feat_dim=feat_dim,
            num_genes=num_genes,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            add_type_embed=add_type_embed,
            type_embedding="additive",
        )


class COASTwoemb(_COASTBase):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        feat_dim: int = 1024,
        num_genes: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        add_type_embed: bool = True,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            feat_dim=feat_dim,
            num_genes=num_genes,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            add_type_embed=add_type_embed,
            type_embedding="none",
        )
