"""Patchify raw HCP sphere .gii files into the pretraining .pt layout.

Reads data/pretraining/raw_sphere/*.gii (per-hemisphere ico6_fs_LR shape
files with curv/sulc/thickness darrays), patchifies each onto the ico6/ico4
grid (see models/patching.py), and writes one .pt file per hemisphere --
shape (C, N, V) as expected by HCPPretrainDataset -- split by subject across
data/pretraining/{train,val}/. reorder=False matches pretrain.py's SiT
encoder (global attention, no locality requirement -- see patching.py).

Files that fail to parse (e.g. truncated downloads) are skipped with a
warning rather than aborting the whole run.
"""
import glob
import os
import os.path as osp
import random
import re
import sys
import warnings

import nibabel as nib
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.patching import load_patch_indices, patchify
from models.config import ENCODER_KWARGS

RAW_ROOT = os.environ.get("PREPARE_RAW_ROOT", "data/pretraining/raw_sphere")
OUT_ROOT = os.environ.get("PREPARE_OUT_ROOT", "data/pretraining")
CHANNELS = ["curv", "sulc", "thickness"]
VAL_FRACTION = float(os.environ.get("PREPARE_VAL_FRACTION", 0.2))
# patch grid must match the encoder -- defaults to the shared config's ico_grid
ICO_GRID = int(os.environ.get("PREPARE_ICO_GRID", ENCODER_KWARGS["ico_grid"]))
SEED = 0


def load_gii_features(path):
    """Returns a (len(CHANNELS), num_vertices) float tensor, channels in
    CHANNELS order (matched by darray metadata Name, not file order)."""
    gii = nib.load(path)
    by_name = {d.meta.get("Name"): d.data for d in gii.darrays}
    missing = [c for c in CHANNELS if c not in by_name]
    if missing:
        raise ValueError(f"{path} is missing darray(s) {missing}, has {list(by_name)}")
    return torch.stack([torch.from_numpy(by_name[c]).float() for c in CHANNELS])

# Output: two pt files per subject, one for each hemisphere
# each file is a tensor of shape (C, N, V)
# where C := channels (one for each surface feature)
# N := Number of patches
# V := NUmber of vertices
def main():
    # Get all .gii files
    files = sorted(glob.glob(osp.join(RAW_ROOT, "*.gii")))
    if not files:
        raise FileNotFoundError(f"no .gii files found under {RAW_ROOT}")

    # Group by subject: for each of them, we have two files (one per hemisphere)
    by_subject = {}
    for f in files:
        m = re.search(r"(sub-\d+)\.([LR])\.", osp.basename(f))
        by_subject.setdefault(m.group(1), []).append((m.group(2), f))

    # Split data in training and validation set
    subjects = sorted(by_subject)
    random.Random(SEED).shuffle(subjects)
    num_val = max(1, round(len(subjects) * VAL_FRACTION))
    val_subjects = set(subjects[:num_val])

    patch_indices = load_patch_indices(ico_mesh=6, ico_grid=ICO_GRID, reorder=False)

    for split_dir in ("train", "val"):
        os.makedirs(osp.join(OUT_ROOT, split_dir), exist_ok=True)

    n_written, n_skipped = 0, 0
    for subject, hemis in sorted(by_subject.items()):
        split = "val" if subject in val_subjects else "train"
        # Patchify hemisphere separately
        for hemi, path in hemis:
            try:
                features = load_gii_features(path)  # (C, V)
            except Exception as e:
                warnings.warn(f"skipping {path}: {e}")
                n_skipped += 1
                continue

            patches = patchify(features, patch_indices)  # (C, N, V)
            out_name = f"{subject}.{hemi}.ico6_fs_LR.pt"
            torch.save(patches, osp.join(OUT_ROOT, split, out_name))
            n_written += 1

    print(f"wrote {n_written} .pt files ({len(subjects) - len(val_subjects)} train subjects, "
          f"{len(val_subjects)} val subjects), skipped {n_skipped}")


if __name__ == "__main__":
    main()
