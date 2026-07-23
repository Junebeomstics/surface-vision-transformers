"""Masked patch prediction pretraining for SiT.

SimMIM-style: a random subset of patches is replaced by a learned mask
token before patch embedding; the encoder runs as normal; a single Linear
decoder reconstructs each masked token's patch content directly (SiT keeps
one token per patch throughout -- no patch merging to invert, unlike
MS-SiT); the loss is the reconstruction MSE restricted to the masked
patches.

This is a simplification of the reference repo's own `models/mpp.py`, which
additionally does randomized patch swapping/replacement scheduling on top of
masking -- kept simple here, same as the pre-existing masking scheme in this
codebase.
"""
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import SiTEncoder

# Random sampling of masks to be masked
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

        # Define a randomly initialized, learnable mask token
        # that will replace masked patches
        patch_dim = encoder.num_vertices * encoder.num_channels
        self.mask_token = nn.Parameter(torch.zeros(1, 1, patch_dim))
        nn.init.trunc_normal_(self.mask_token, std=.02)

        # Reconstruct patch from mask token embedding
        self.decoder = nn.Linear(encoder.embed_dim, patch_dim)

    def forward(self, x):
        # x: (B, C, N, V) -> flat patches (B, N, V*C), same layout the encoder embeds
        patches = self.encoder.to_patch_embedding[0](x)
        mask = random_mask(patches.shape[0], patches.shape[1], self.mask_ratio, x.device)

        # Mask out some patches with learnable mask token
        corrupted = torch.where(mask.unsqueeze(-1), self.mask_token, patches)
        embedded = self.encoder.to_patch_embedding[1](corrupted)

        tokens = embedded + self.encoder.pos_embedding
        tokens = self.encoder.dropout(tokens)
        for block in self.encoder.blocks:
            tokens = block(tokens)
        tokens = self.encoder.norm(tokens)

        reconstruction = self.decoder(tokens)
        loss = F.mse_loss(reconstruction[mask], patches[mask])
        return loss, reconstruction, mask


class SiTPretraining(L.LightningModule):
    """LightningModule wrapping masked patch prediction pretraining of the
    SiT encoder. `self.encoder` holds the weights to carry over to
    downstream fine-tuning."""

    def __init__(self, encoder_kwargs, mask_ratio=0.6, lr=1e-4, weight_decay=0.05):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = SiTEncoder(**encoder_kwargs)
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
