"""
Masked surface modelling (MAE-style self-supervision) for the MS-SiT backbone.

The existing MPP implementation in `models/mpp.py` cannot be reused for MS-SiT:
it reaches into SiT internals (`cls_token`, `pos_embedding`, `transformer`,
`dropout`) that the hierarchical MS-SiT does not define, and it assumes a 1:1
token-to-patch correspondence that MS-SiT breaks through patch merging.

This module takes the other route. MS-SiT already ships a U-Net variant
(`models/ms_sit_unet.py`) whose decoder returns to full patch resolution and
projects back onto the 40,962-vertex sphere. Setting `num_classes = num_channels`
turns it into a reconstruction network, so pretraining reduces to:

    1. corrupt a random subset of input patches with a learnable mask token
    2. run the U-Net
    3. compute the reconstruction loss over masked vertices only

Masking happens in input space (before `to_patch_embedding`), matching what
`mpp.py` does for SiT, so the backbone itself needs no modification.

PATCH OVERLAP. Neighbouring triangular patches share their edges and corners,
so 5120 patches x 15 vertices covers 40,962 vertices with 76,800 slots:

    coverage 1 patch  : 15,360 vertices (37.50%)  patch interior
    coverage 2 patches: 23,040 vertices (56.25%)  patch edges
    coverage 5-6      :  2,562 vertices ( 6.26%)  patch corners

This matters for the loss mask. A vertex is hidden from the network only when
every patch covering it is masked; if any covering patch survives, the value is
still literally present in the input. Scoring the loss on the union instead
would make 43% of the loss terms (at mask_prob 0.6) trivially satisfiable by
copying the input. `masked_vertices` therefore uses the intersection, and the
hidden-vertex fraction runs well below the patch masking rate:

    mask_prob   0.50   0.60   0.75   0.85
    hidden      0.33   0.43   0.61   0.74

so mask_prob 0.85 is what corresponds to the usual 75% MAE masking here.

LIMITATION - partial weight transfer. `ms_sit.py` implements window attention
with a learned `relative_position_bias_table` per block; `ms_sit_unet.py` does
not. Pretraining therefore covers 159 of the 185 tensors in a downstream MSSiT:
everything except the task head (2) and the 12 blocks' relative position bias
tables and index buffers (24). Those 12 tables start from scratch at finetuning
time. Aligning the two attention implementations would close the gap.
"""

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

N_VERTICES_ICO6 = 40962


def patch_to_vertex_index(indices_df, num_patches):
    """(num_patches, num_vertices) int64 tensor from the triangle-indices frame.

    `indices_df` is the DataFrame the MS-SiT models load from
    patch_extraction/triangle_indices_ico_6_sub_ico_<n>.csv, already reordered
    if the model was built with reorder=True. Columns are patch ids as strings.
    """
    cols = [indices_df[str(i)].values for i in range(num_patches)]
    return torch.from_numpy(np.stack(cols, axis=0).astype(np.int64))


class masked_surface_modelling(nn.Module):
    """MAE-style pretraining wrapper around an MS-SiT U-Net.

    Args:
        transformer: an MSSiTUNet built with num_classes == num_channels.
        patch_to_vertex: (num_patches, num_vertices) long tensor, see above.
        mask_prob: fraction of patches corrupted per sample.
        replace_prob: of the corrupted patches, the fraction replaced by the
            mask token. The remainder is swapped with a random patch from the
            same sample, which keeps the input distribution closer to real data
            (same idea as the swap term in mpp.py).
        norm_target: standardise the reconstruction target per sample and
            channel over masked vertices. Follows the normalised-pixel target of
            He et al. and keeps channels with different units (curv ~1e-1 vs
            thickness ~mm) from dominating the loss.
    """

    def __init__(self,
                 transformer,
                 patch_to_vertex,
                 num_channels,
                 mask_prob=0.5,
                 replace_prob=0.8,
                 norm_target=True,
                 n_vertices=N_VERTICES_ICO6):
        super().__init__()

        self.transformer = transformer
        self.num_channels = num_channels
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.norm_target = norm_target
        self.n_vertices = n_vertices

        self.num_patches, self.num_vertices = patch_to_vertex.shape
        self.register_buffer("patch_to_vertex", patch_to_vertex, persistent=False)
        # flat scatter index, reused every step
        self.register_buffer("flat_index", patch_to_vertex.reshape(-1), persistent=False)

        # How many patches cover each vertex. For ico6/ico4 this is 1 for the
        # 15,360 patch-interior vertices, 2 for the 23,040 edge vertices and
        # 5-6 for the 2,562 corner vertices. Needed to tell a genuinely hidden
        # vertex from one that merely sits in a masked patch.
        coverage = torch.bincount(patch_to_vertex.reshape(-1), minlength=n_vertices)
        self.register_buffer("coverage", coverage.clamp(min=1), persistent=False)

        self.mask_token = nn.Parameter(
            torch.randn(1, 1, self.num_vertices * num_channels) * 0.02
        )

    # ------------------------------------------------------------------ #

    def scatter_to_sphere(self, x):
        """(B,C,L,V) patches -> (B,C,n_vertices) sphere.

        Mirrors MSSiTUNet.repatch: later patches win on shared vertices.
        """
        b, c = x.shape[0], x.shape[1]
        src = rearrange(x, "b c n v -> b c (n v)")
        out = torch.zeros(b, c, self.n_vertices, device=x.device, dtype=x.dtype)
        out[:, :, self.flat_index] = src
        return out

    def masked_vertices(self, patch_mask):
        """(B,L) bool patch mask -> (B,n_vertices) bool "genuinely hidden" mask.

        A vertex is hidden only when *every* patch covering it is masked. Using
        the union instead (any covering patch masked) would put vertices into
        the loss that are still literally present in the input via an unmasked
        neighbouring patch: at mask_prob 0.6 that is 43% of the loss terms,
        which collapses the pretext task into partly copying the input.
        """
        b = patch_mask.shape[0]
        counts = torch.zeros(b, self.n_vertices, device=patch_mask.device)
        contrib = patch_mask.float().repeat_interleave(self.num_vertices, dim=1)
        counts.scatter_add_(1, self.flat_index.unsqueeze(0).expand(b, -1), contrib)
        return counts >= self.coverage.unsqueeze(0)

    def corrupt(self, batch, patch_mask):
        """Replace masked patches with the mask token or a random patch."""
        corrupted = batch.clone()

        use_token = (torch.rand_like(patch_mask, dtype=torch.float) < self.replace_prob) & patch_mask
        corrupted[use_token] = self.mask_token.to(batch.dtype).squeeze(0).squeeze(0)

        swap = patch_mask & ~use_token
        if swap.any():
            b, n, _ = batch.shape
            donor = torch.randint(0, n, (b, n), device=batch.device)
            donated = torch.gather(
                batch, 1, donor.unsqueeze(-1).expand(-1, -1, batch.shape[-1])
            )
            corrupted[swap] = donated[swap]
        return corrupted

    # ------------------------------------------------------------------ #

    def forward(self, x, confounds=None):
        """x: (B, C, L, V). Returns (loss, dict of diagnostics)."""
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"expected {self.num_channels} input channels, got {x.shape[1]}"
            )

        target = self.scatter_to_sphere(x)                     # B,C,n_vertices
        batch = rearrange(x, "b c n v -> b n (v c)")            # B,L,V*C

        patch_mask = torch.rand(
            batch.shape[0], self.num_patches, device=x.device
        ) < self.mask_prob
        # guarantee at least one masked patch so the loss is always defined
        if not patch_mask.any():
            patch_mask[:, 0] = True

        corrupted = self.corrupt(batch, patch_mask)
        corrupted = rearrange(corrupted, "b n (v c) -> b c n v", v=self.num_vertices)

        pred = self.transformer(corrupted, confounds) if confounds is not None \
            else self.transformer(corrupted)                    # B,C,n_vertices

        vmask = self.masked_vertices(patch_mask)                # B,n_vertices
        vmask_c = vmask.unsqueeze(1).expand_as(target)

        if self.norm_target:
            target = self._normalise_target(target, vmask)

        diff = (pred - target) ** 2
        loss = diff[vmask_c].mean()

        stats = {
            "masked_patch_frac": patch_mask.float().mean().item(),
            "hidden_vertex_frac": vmask.float().mean().item(),
        }
        return loss, stats

    def _normalise_target(self, target, vmask):
        """Per-sample, per-channel standardisation over masked vertices."""
        m = vmask.unsqueeze(1).to(target.dtype)                 # B,1,V
        n = m.sum(dim=2, keepdim=True).clamp(min=1)
        mean = (target * m).sum(dim=2, keepdim=True) / n
        var = (((target - mean) ** 2) * m).sum(dim=2, keepdim=True) / n
        return (target - mean) / (var + 1e-6).sqrt()
