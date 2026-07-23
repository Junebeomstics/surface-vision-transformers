"""Building blocks for the SiT encoder. Ported from the reference
implementation: https://github.com/metrics-lab/surface-vision-transformers
(models/sit.py, which wraps vit_pytorch.vit.Transformer).
"""
import torch.nn as nn


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


class Attention(nn.Module):
    """Standard global multi-head self-attention (no relative position bias --
    SiT relies entirely on the encoder's absolute positional embedding)."""

    def __init__(self, dim, num_heads, dim_head=64, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = (q * self.scale) @ k.transpose(-2, -1)
        attention = self.attn_drop(self.softmax(attention))

        x = (attention @ v).transpose(1, 2).reshape(B, N, self.num_heads * self.dim_head)
        return self.proj_drop(self.proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm residual block: x = x + attn(norm1(x)); x = x + mlp(norm2(x))."""

    def __init__(self, dim, num_heads, dim_head=64, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attention = Attention(dim, num_heads, dim_head=dim_head, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))
