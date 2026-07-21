"""SiT with a per-token linear head for dense (per-vertex) prediction.

The SiT paper (https://arxiv.org/abs/2203.16414) only defines a
classification/global-regression head (cls-token pooling -> single value).
This project needs one value per mesh vertex instead, which the paper doesn't
cover, so the head here is our own: since SiT keeps every patch as a token
throughout the encoder (no hierarchy to invert, unlike MS-SiT's U-Net-style
decoder), each output token is expanded back to its patch's vertices with a
single Linear, then unpatched onto the full mesh (see patching.py) and
projected to per-vertex predictions.

This is the architecture src/finetuning uses for retinotopy: dense regression
over every mesh vertex, with the loss restricted to the ROI at training time
(see finetuning/sit_dataset.py).
"""
import torch.nn as nn
from einops import rearrange

from .encoder import SiTEncoder
from .patching import DEFAULT_ROOT, load_patch_indices, unpatchify_mean


class SiTDense(nn.Module):
    def __init__(self, encoder_kwargs, num_classes=1, ico_mesh=6, reorder=False,
                 patch_root=DEFAULT_ROOT, norm_layer=nn.LayerNorm):
        super().__init__()
        self.encoder = SiTEncoder(**encoder_kwargs)
        self.num_classes = num_classes

        embed_dim = self.encoder.embed_dim

        # per-patch-token -> per-vertex-within-patch expansion, then unpatch to mesh, then predict
        self.up = nn.Linear(embed_dim, embed_dim * self.encoder.num_vertices)
        patch_indices = load_patch_indices(ico_mesh=ico_mesh, ico_grid=self._ico_grid_from(encoder_kwargs),
                                            reorder=reorder, root=patch_root)
        self.register_buffer("patch_indices", patch_indices)
        self.num_mesh_vertices = int(patch_indices.max().item()) + 1
        self.final_norm = norm_layer(embed_dim)
        self.output = nn.Linear(embed_dim, num_classes)

    @staticmethod
    def _ico_grid_from(encoder_kwargs):
        return encoder_kwargs.get("ico_grid", 4)

    def unpatch_to_mesh(self, x):
        # x: (B, N, embed_dim) -> (B, N, V, embed_dim)
        x = self.up(x)
        x = rearrange(x, 'b n (v c) -> b n v c', v=self.encoder.num_vertices)
        # unpatchify_mean expects (..., N, V); move the channel dim out front
        x = rearrange(x, 'b n v c -> b c n v')
        x = unpatchify_mean(x, self.patch_indices, self.num_mesh_vertices)  # B, C, num_mesh_vertices
        return rearrange(x, 'b c m -> b m c')

    def forward(self, x):
        # x: (B, C, N, V)
        tokens = self.encoder(x)
        mesh_features = self.unpatch_to_mesh(tokens)  # B, num_mesh_vertices, embed_dim
        mesh_features = self.final_norm(mesh_features)
        out = self.output(mesh_features)  # B, num_mesh_vertices, num_classes
        return out.squeeze(-1) if self.num_classes == 1 else out.transpose(1, 2)
