"""MS-SiT with a symmetric decoder for dense (per-vertex) prediction.

Adapted from the reference implementation's segmentation model
(models/ms_sit_unet.py): the encoder's hierarchical downsampling is mirrored
by an upsampling decoder (PatchExpand, the inverse of PatchMerging) with
U-Net-style skip connections from each encoder stage. The decoder's output
tokens are then unpatched back onto mesh vertices (see patching.py) and
projected to per-vertex predictions.

This is the architecture src/finetuning uses for retinotopy: dense
regression over every mesh vertex, with the loss restricted to the ROI at
training time (see finetuning/mssit_dataset.py).
"""
import torch
import torch.nn as nn
from einops import rearrange

from .encoder import MSSiTEncoder
from .layers import BasicLayer, PatchExpand
from .patching import DEFAULT_ROOT, load_patch_indices, unpatchify_mean


class MSSiTDense(nn.Module):
    def __init__(self, encoder_kwargs, num_classes=1, ico_mesh=6, reorder=True,
                 patch_root=DEFAULT_ROOT, norm_layer=nn.LayerNorm):
        super().__init__()
        self.encoder = MSSiTEncoder(**encoder_kwargs)
        self.num_classes = num_classes

        num_layers = self.encoder.num_layers
        embed_dim = self.encoder.embed_dim
        depths = [len(layer.blocks) for layer in self.encoder.layers]
        num_heads = [layer.blocks[0].attention.num_heads for layer in self.encoder.layers]
        window_sizes = self.encoder.window_sizes

        self.layers_up = nn.ModuleList()
        self.skip_proj = nn.ModuleList()
        for ind in range(num_layers):
            dim = int(embed_dim * 2 ** (num_layers - 1 - ind))
            if ind == 0:
                self.layers_up.append(PatchExpand(hidden_dim=self.encoder.num_features, norm_layer=norm_layer))
                self.skip_proj.append(nn.Identity())
            else:
                stage_idx = num_layers - 1 - ind
                self.layers_up.append(BasicLayer(
                    hidden_dim=dim, depth=depths[stage_idx], num_heads=num_heads[stage_idx],
                    window_size=window_sizes[stage_idx], norm_layer=norm_layer,
                    resample=PatchExpand if ind < num_layers - 1 else None,
                ))
                self.skip_proj.append(nn.Linear(2 * dim, dim))

        self.norm_up = norm_layer(embed_dim)

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

    def forward_up_features(self, x, skips):
        num_layers = len(self.layers_up)
        for ind, layer_up in enumerate(self.layers_up):
            if ind == 0:
                x = layer_up(x)
            else:
                skip = skips[num_layers - 1 - ind]
                x = self.skip_proj[ind](torch.cat([x, skip], dim=-1))
                x = layer_up(x)
        return self.norm_up(x)

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
        bottleneck, skips = self.encoder(x, return_skips=True)
        tokens = self.forward_up_features(bottleneck, skips)
        mesh_features = self.unpatch_to_mesh(tokens)  # B, num_mesh_vertices, embed_dim
        mesh_features = self.final_norm(mesh_features)
        out = self.output(mesh_features)  # B, num_mesh_vertices, num_classes
        return out.squeeze(-1) if self.num_classes == 1 else out.transpose(1, 2)
