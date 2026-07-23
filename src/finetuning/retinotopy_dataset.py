"""Retinotopy fine-tuning dataset over full-hemisphere SiT inputs.

Reads the per-subject caches built by prepare_finetuning.py (input already
patchified to the icosphere grid; labels on the full ico6 mesh). One sample =
one subject-hemisphere: the whole-hemisphere SiT input plus the full-mesh
target / R2 / eccentricity. Supervision and evaluation are restricted to the
retinotopy ROI downstream (see fine_tune.py, which owns the ROI mask and the
R2>threshold / EVC / ecc<=12 evaluation policy) -- the dataset stays policy-free
and just serves tensors.

Returns (x, y, r2, ecc):
    x   : (C, N, V) patchified curv/sulc/thickness -- same features as pretraining
    y   : (num_mesh_vertices,) target for `prediction`
    r2  : (num_mesh_vertices,) pRF variance explained (0-100 scale)
    ecc : (num_mesh_vertices,) eccentricity, for the ecc<=12 evaluation mask
"""
import os.path as osp

import torch
from torch.utils.data import Dataset

_HEMI = {"Left": "L", "Right": "R"}


def read_subjects(split_file):
    """Six-digit subject ids, one per line (RetinoSolver subject_splits format)."""
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


class RetinotopyDataset(Dataset):
    def __init__(self, cache_root, subjects, hemisphere="Left", prediction="polarAngle"):
        hemi = _HEMI[hemisphere]
        self.prediction = prediction
        self.paths = [osp.join(cache_root, f"sub-{s}.{hemi}.pt") for s in subjects]
        self.paths = [p for p in self.paths if osp.exists(p)]
        if not self.paths:
            raise FileNotFoundError(
                f"no cached subjects found in {cache_root} for hemisphere {hemisphere}; "
                "run src/finetuning/prepare_finetuning.py first"
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        d = torch.load(self.paths[idx], weights_only=True)
        return d["x"], d[self.prediction], d["R2"], d["eccentricity"]
