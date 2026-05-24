from __future__ import annotations

import torch
from torch import nn

from .common import masked_mean


class FullAuxRescorer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 29,
        token_embed_dim: int = 128,
        precursor_dim: int = 64,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, token_embed_dim, padding_idx=0)

        token_layer = nn.TransformerEncoderLayer(
            d_model=token_embed_dim,
            nhead=n_heads,
            dim_feedforward=token_embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.token_encoder = nn.TransformerEncoder(token_layer, num_layers=n_layers)

        self.spec_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        spec_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.spec_encoder = nn.TransformerEncoder(spec_layer, num_layers=n_layers)

        self.precursor_mlp = nn.Sequential(
            nn.Linear(2, precursor_dim),
            nn.GELU(),
            nn.LayerNorm(precursor_dim),
            nn.GELU(),
            nn.Linear(precursor_dim, precursor_dim),
        )
        self.aux_feature_mlp = nn.Sequential(
            nn.Linear(10, precursor_dim),
            nn.GELU(),
            nn.LayerNorm(precursor_dim),
            nn.GELU(),
            nn.Linear(precursor_dim, precursor_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(token_embed_dim + hidden_dim + precursor_dim + precursor_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        spectra: torch.Tensor,
        spectra_mask: torch.Tensor,
        precursors: torch.Tensor,
        tokens: torch.Tensor,
        aux_features: torch.Tensor,
    ) -> torch.Tensor:
        token_feat = self.encode_token_branch(tokens)
        spec_feat = self.encode_spectrum_branch(spectra, spectra_mask)
        return self.score_from_branch_features(token_feat, spec_feat, precursors, aux_features)

    def encode_token_branch(self, tokens: torch.Tensor) -> torch.Tensor:
        token_pad_mask = tokens.eq(0)
        token_hidden = self.token_encoder(
            self.token_embed(tokens),
            src_key_padding_mask=token_pad_mask,
        )
        return masked_mean(token_hidden, token_pad_mask)

    def encode_spectrum_branch(
        self,
        spectra: torch.Tensor,
        spectra_mask: torch.Tensor,
    ) -> torch.Tensor:
        spec_hidden = self.spec_encoder(
            self.spec_proj(spectra),
            src_key_padding_mask=spectra_mask.bool(),
        )
        return masked_mean(spec_hidden, spectra_mask.bool())

    def score_from_branch_features(
        self,
        token_feat: torch.Tensor,
        spec_feat: torch.Tensor,
        precursors: torch.Tensor,
        aux_features: torch.Tensor,
    ) -> torch.Tensor:
        precursor_feat = self.precursor_mlp(precursors)
        aux_feat = self.aux_feature_mlp(aux_features)
        fused = torch.cat([token_feat, spec_feat, precursor_feat, aux_feat, precursors], dim=1)
        return self.classifier(fused).squeeze(-1)
