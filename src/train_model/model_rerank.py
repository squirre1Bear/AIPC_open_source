import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class AIPCRerankNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_embed_dim: int = 128,
        precursor_dim: int = 64,
        hidden_dim: int = 256,
        aux_feature_dim: int = 10,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_token_len: int = 64,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, token_embed_dim, padding_idx=0)
        self.token_pos = PositionalEncoding(token_embed_dim, max_len=max_token_len)

        token_layer = nn.TransformerEncoderLayer(
            d_model=token_embed_dim,
            nhead=max(1, min(n_heads, token_embed_dim // 16 if token_embed_dim >= 16 else 1)),
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
            nn.Dropout(dropout),
            nn.Linear(precursor_dim, precursor_dim),
            nn.GELU(),
        )

        self.aux_feature_mlp = nn.Sequential(
            nn.Linear(aux_feature_dim, precursor_dim),
            nn.GELU(),
            nn.LayerNorm(precursor_dim),
            nn.Dropout(dropout),
            nn.Linear(precursor_dim, precursor_dim),
            nn.GELU(),
        )

        fused_dim = hidden_dim + token_embed_dim + precursor_dim + precursor_dim + 2
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        valid = (~mask).unsqueeze(-1).float()
        denom = valid.sum(dim=1).clamp_min(eps)
        return (x * valid).sum(dim=1) / denom

    def forward(self, spectra, spectra_mask, precursors, tokens, aux_features=None):
        token_pad_mask = tokens.eq(0)
        tok = self.token_embed(tokens)
        tok = self.token_pos(tok)
        tok = self.token_encoder(tok, src_key_padding_mask=token_pad_mask)
        tok_pool = self.masked_mean(tok, token_pad_mask)

        spec = self.spec_proj(spectra)
        spec = self.spec_encoder(spec, src_key_padding_mask=spectra_mask)
        spec_pool = self.masked_mean(spec, spectra_mask)

        prec = self.precursor_mlp(precursors[:, :2])
        if aux_features is None:
            aux_features = torch.zeros(
                (precursors.size(0), self.aux_feature_mlp[0].in_features),
                dtype=precursors.dtype,
                device=precursors.device,
            )
        aux = self.aux_feature_mlp(aux_features)

        tok_for_cos = tok_pool
        if tok_for_cos.size(1) < spec_pool.size(1):
            tok_for_cos = F.pad(tok_for_cos, (0, spec_pool.size(1) - tok_for_cos.size(1)))
        else:
            tok_for_cos = tok_for_cos[:, :spec_pool.size(1)]

        cosine = F.cosine_similarity(
            F.normalize(spec_pool, dim=-1),
            F.normalize(tok_for_cos, dim=-1),
            dim=-1,
        ).unsqueeze(-1)

        length_feat = tokens.ne(0).sum(dim=1, keepdim=True).float() / max(1, tokens.size(1))

        fused = torch.cat([spec_pool, tok_pool, prec, aux, cosine, length_feat], dim=-1)
        logit = self.classifier(fused).squeeze(-1)
        return logit
