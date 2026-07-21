"""MS-SiT encoder (Multiscale Surface Vision Transformer).

Adapted and trimmed from the reference implementation:
https://github.com/metrics-lab/surface-vision-transformers (models/ms_sit.py)

A Swin-style hierarchical transformer over icosphere patches. Input tensors
have shape (B, C, N, V): batch, channels, number of patches, vertices per
patch. Patches are merged 4-to-1 between stages (like Swin windows), so the
token sequence shrinks N -> N/4 -> N/16 -> ... across stages while the
channel dimension doubles. Used as-is for masked-patch pretraining
(src/pretraining) and as the backbone inside MSSiTDense for fine-tuning
(src/finetuning).
"""
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from timm.layers import trunc_normal_

from .layers import BasicLayer, PatchMerging

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


class MSSiTEncoder(nn.Module):
    def __init__(self, ico_grid=4, num_channels=4,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=(64, 64, 64, 80), mlp_ratio=4, qkv_bias=True, qk_scale=None,
                 dropout=0., attention_dropout=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, use_pos_emb=False):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.use_pos_emb = use_pos_emb
        self.num_channels = num_channels
        # channel width of the final stage, after num_layers-1 doublings
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.window_sizes = [window_size] * self.num_layers if isinstance(window_size, int) else list(window_size)

        self.num_patches, self.num_vertices = ICO_GRID[ico_grid]
        patch_dim = self.num_vertices * self.num_channels

        if use_pos_emb:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v -> b n (v c)'),
            nn.Linear(patch_dim, self.embed_dim),
        )

        self.pos_dropout = nn.Dropout(p=dropout)

        # stochastic depth decay rule, one rate per block across all stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(BasicLayer(
                hidden_dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_sizes[i_layer],
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=dropout, attn_drop=attention_dropout,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                resample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            ))

        self.pre_norm = norm_layer(self.embed_dim)
        self.norm = norm_layer(self.num_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_skips=False):
        """x: (B, C, N, V). Returns the final-stage token sequence
        (B, N // 4**(num_layers-1), num_features). With return_skips=True,
        also returns each stage's input tokens (pre-merge), for use as
        U-Net-style skip connections by a decoder."""
        tokens = self.to_patch_embedding(x)
        tokens = self.pre_norm(tokens)
        if self.use_pos_emb:
            tokens = tokens + self.absolute_pos_embed
        tokens = self.pos_dropout(tokens)

        skips = []
        for layer in self.layers:
            skips.append(tokens)
            tokens = layer(tokens)
        tokens = self.norm(tokens)

        return (tokens, skips) if return_skips else tokens
