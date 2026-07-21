"""MS-SiT encoder (Multiscale Surface Vision Transformer).

Adapted and trimmed from the reference implementation:
https://github.com/metrics-lab/surface-vision-transformers (models/ms_sit.py)

A Swin-style hierarchical transformer over icosphere patches. Input tensors
have shape (B, C, N, V): batch, channels, number of patches, vertices per
patch. Patches are merged 4-to-1 between stages (like Swin windows), so the
token sequence shrinks N -> N/4 -> N/16 -> ... across stages while the
channel dimension doubles.
"""
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from timm.layers import DropPath, trunc_normal_

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


class MSSiT(nn.Module):
    def __init__(self, ico_grid=4, num_channels=4, num_classes=1,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=(64, 64, 64, 80), mlp_ratio=4, qkv_bias=True, qk_scale=None,
                 dropout=0., attention_dropout=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, use_pos_emb=False):
        super().__init__()

        self.num_classes = num_classes
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
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            ))

        self.pre_norm = norm_layer(self.embed_dim)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        """Patch embedding + hierarchical stages. Returns the final-stage
        token sequence (B, N // 4**(num_layers-1), num_features), i.e. the
        representation before global pooling."""
        x = self.to_patch_embedding(x)  # B, N, embed_dim
        x = self.pre_norm(x)
        if self.use_pos_emb:
            x = x + self.absolute_pos_embed
        x = self.pos_dropout(x)

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)

    def forward_features(self, x):
        """Pooled representation (B, num_features), for classification/regression heads."""
        x = self.forward_encoder(x)
        x = self.avgpool(x.transpose(1, 2))
        return torch.flatten(x, 1)

    def forward(self, x):
        return self.head(self.forward_features(x))


class BasicLayer(nn.Module):
    """One Swin-style stage: a stack of windowed-attention blocks, optionally
    followed by patch merging (4 tokens -> 1, channels x2)."""

    def __init__(self, hidden_dim, depth, num_heads, window_size,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                hidden_dim=hidden_dim, num_heads=num_heads, window_size=window_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.downsample = downsample(hidden_dim=hidden_dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class SwinTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, window_size=80,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(hidden_dim)
        self.attention = WindowAttention(hidden_dim, window_size, num_heads,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(hidden_dim)
        self.mlp = MLP(hidden_dim, int(hidden_dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x_windows = window_partition(x, self.window_size)
        attention_windows, _ = self.attention(x_windows)
        x = window_reverse(attention_windows, self.window_size, L)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def window_partition(x, window_size):
    """(B, L, C) -> (B * L//window_size, window_size, C)"""
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    return x.permute(0, 2, 1, 3).contiguous().view(-1, window_size, C)


def window_reverse(windows, window_size, L):
    B = int(windows.shape[0] / (L // window_size))
    x = windows.view(B, L // window_size, window_size, -1)
    return x.contiguous().view(B, L, -1)


class WindowAttention(nn.Module):
    def __init__(self, hidden_dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = hidden_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # relative position bias over the 1-D window
        self.relative_position_bias_table = nn.Parameter(torch.zeros(2 * window_size - 1, num_heads))
        coords = torch.arange(self.window_size)
        relative_coords = coords[:, None] - coords[None, :] + self.window_size - 1
        self.register_buffer("relative_position_index", relative_coords)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        Bw, L, C = x.shape
        qkv = self.qkv(x).reshape(Bw, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = (q * self.scale) @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(self.window_size, self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention = attention + relative_position_bias.unsqueeze(0)

        attention = self.attn_drop(self.softmax(attention))
        x = (attention @ v).transpose(1, 2).reshape(Bw, L, C)
        x = self.proj_drop(self.proj(x))
        return x, attention


class PatchMerging(nn.Module):
    """Merges groups of 4 consecutive tokens into 1, doubling channel width."""

    def __init__(self, hidden_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.merging = Rearrange('b (v h) n -> b v (h n)', h=4)
        self.norm = norm_layer(4 * hidden_dim)
        self.reduction = nn.Linear(4 * hidden_dim, 2 * hidden_dim, bias=False)

    def forward(self, x):
        return self.reduction(self.norm(self.merging(x)))
