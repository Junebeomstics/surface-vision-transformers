"""HCP pretraining data loading.

Expects pre-patched cortical surface data, one .pt file per sample, laid
out as:

    data/pretraining/train/*.pt
    data/pretraining/val/*.pt

Each .pt file holds a single float tensor of shape (C, N, V): channels
(e.g. curvature, myelination, thickness, sulcal depth), number of
patches, and vertices per patch -- i.e. already patched onto the
icosphere grid with the reference repo's `patch_extraction` pipeline, in
the same layout MSSiT.forward expects (batched to B, C, N, V). No labels
are needed since pretraining is self-supervised.
"""
import os.path as osp
from glob import glob

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset


class HCPPretrainDataset(Dataset):
    def __init__(self, root, split):
        self.files = sorted(glob(osp.join(root, split, "*.pt")))
        if not self.files:
            raise FileNotFoundError(f"no .pt files found under {osp.join(root, split)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=True).float()


class HCPDataModule(L.LightningDataModule):
    def __init__(self, root="data/pretraining", batch_size=16, num_workers=8):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_ds = HCPPretrainDataset(self.hparams.root, "train")
        self.val_ds = HCPPretrainDataset(self.hparams.root, "val")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True,
                           num_workers=self.hparams.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, shuffle=False,
                           num_workers=self.hparams.num_workers)
