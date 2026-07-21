"""Masked patch prediction pretraining for MS-SiT.

SimMIM-style: a random subset of patches is replaced by a learned mask
token before patch embedding; the encoder (all hierarchical stages) runs
as normal; a lightweight linear decoder expands the final-stage token
sequence back to full patch resolution; the loss is the reconstruction
MSE restricted to the masked patches.

The reference repo's own `models/mpp.py` only targets the flat (non
hierarchical) SiT model -- it masks patches and feeds them through a
single-resolution transformer, which doesn't apply to MS-SiT since its
Swin-style patch merging shrinks the token sequence between stages
(N -> N/4 -> N/16 -> ...). The decoder below inverts exactly that
merging: each final-stage token corresponds to a contiguous run of
`expansion = 4**(num_layers-1)` original patches (see `PatchMerging` in
model.py), so a single Linear expanding each token to
`expansion * patch_dim` values, reshaped back into the sequence, is a
valid full-resolution reconstruction head.
"""
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from model import MSSiT


def random_mask(batch_size, num_patches, mask_ratio, device):
    """Boolean mask (B, N), exactly round(mask_ratio * N) True per row."""
    num_masked = max(1, int(mask_ratio * num_patches))
    noise = torch.rand(batch_size, num_patches, device=device)
    masked_idx = noise.topk(num_masked, dim=-1).indices
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    mask.scatter_(1, masked_idx, True)
    return mask


class MaskedPatchPrediction(nn.Module):
    def __init__(self, encoder, mask_ratio=0.6):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio

        patch_dim = encoder.num_vertices * encoder.num_channels
        self.mask_token = nn.Parameter(torch.zeros(1, 1, patch_dim))
        nn.init.trunc_normal_(self.mask_token, std=.02)

        self.expansion = 4 ** (encoder.num_layers - 1)
        assert encoder.num_patches % self.expansion == 0
        self.decoder = nn.Sequential(
            nn.Linear(encoder.num_features, self.expansion * patch_dim),
            Rearrange('b v (h p) -> b (v h) p', h=self.expansion),
        )

    def forward(self, x):
        # x: (B, C, N, V) -> flat patches (B, N, V*C), same layout the encoder embeds
        patches = self.encoder.to_patch_embedding[0](x)
        mask = random_mask(patches.shape[0], patches.shape[1], self.mask_ratio, x.device)

        corrupted = torch.where(mask.unsqueeze(-1), self.mask_token, patches)
        embedded = self.encoder.to_patch_embedding[1](corrupted)

        tokens = self.encoder.pre_norm(embedded)
        if self.encoder.use_pos_emb:
            tokens = tokens + self.encoder.absolute_pos_embed
        tokens = self.encoder.pos_dropout(tokens)
        for layer in self.encoder.layers:
            tokens = layer(tokens)
        tokens = self.encoder.norm(tokens)

        reconstruction = self.decoder(tokens)
        loss = F.mse_loss(reconstruction[mask], patches[mask])
        return loss, reconstruction, mask


class MSSiTPretraining(L.LightningModule):
    """LightningModule wrapping masked patch prediction pretraining of the
    MS-SiT encoder. `self.encoder` holds the weights to carry over to
    downstream fine-tuning."""

    def __init__(self, encoder_kwargs, mask_ratio=0.6, lr=1e-4, weight_decay=0.05):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = MSSiT(num_classes=0, **encoder_kwargs)
        self.mpp = MaskedPatchPrediction(self.encoder, mask_ratio=mask_ratio)

    def training_step(self, batch, _):
        loss, _, _ = self.mpp(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch.shape[0], sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        loss, _, _ = self.mpp(batch)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.shape[0], sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
