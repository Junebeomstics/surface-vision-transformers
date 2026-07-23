"""SiT encoder (Surface Vision Transformer, non-multiscale).

Adapted from the reference implementation:
https://github.com/metrics-lab/surface-vision-transformers (models/sit.py)

A flat ViT over icosphere patches: global self-attention, no hierarchy/patch
merging. Input tensors have shape (B, C, N, V): batch, channels, number of
patches, vertices per patch. Every patch stays a token throughout the whole
encoder, so the output sequence is (B, N, embed_dim). Used as-is for
masked-patch pretraining (src/pretraining) and as the backbone inside SiTDense
for fine-tuning (src/finetuning).

Unlike the reference SiT, there is no cls token / pooling here: this project
only ever does dense per-vertex prediction, which reads every patch token's
output directly (see models/dense_head.py), so a pooled global token would be
unused dead weight.
"""
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from timm.layers import trunc_normal_

from .layers import TransformerBlock

# number of patches / vertices-per-patch for each icosphere subdivision used
# as the patching grid (sub_ico_N in the reference repo's configs)
ICO_GRID = {
    0: (20, 2145),
    1: (80, 561),
    2: (320, 153),
    3: (1280, 45),
    4: (5120, 15),
    5: (20480, 6),
}


class SiTEncoder(nn.Module):
    def __init__(self, ico_grid=4, num_channels=3,
                 embed_dim=192, depth=12, num_heads=3, dim_head=64, mlp_ratio=4,
                 qkv_bias=True, dropout=0., emb_dropout=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_channels = num_channels

        self.num_patches, self.num_vertices = ICO_GRID[ico_grid]
        patch_dim = self.num_vertices * self.num_channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v -> b n (v c)'),
            nn.Linear(patch_dim, self.embed_dim),
        )

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        trunc_normal_(self.pos_embedding, std=.02)

        self.dropout = nn.Dropout(p=emb_dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, dim_head=dim_head, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=dropout, attn_drop=dropout, norm_layer=norm_layer,
            )
            for _ in range(depth)
        ])

        self.norm = norm_layer(self.embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """x: (B, C, N, V). Returns the token sequence (B, N, embed_dim)."""
        tokens = self.to_patch_embedding(x)
        tokens = tokens + self.pos_embedding
        tokens = self.dropout(tokens)

        for block in self.blocks:
            tokens = block(tokens)

        return self.norm(tokens)
