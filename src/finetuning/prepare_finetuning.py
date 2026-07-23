"""Build per-subject fine-tuning tensors for the retinotopy task.

For every subject-hemisphere that has retinotopy ground truth, bundle the SiT
input (curv/sulc/thickness patchified onto the icosphere grid, identical to the
pretraining input) together with the full-mesh labels (polarAngle, eccentricity,
pRFsize, R2). Writing these once avoids re-parsing gii files on every CV fold.

    input  x : (C, N, V) patchified curv/sulc/thickness  -- same as pretraining
    labels   : polarAngle / eccentricity / pRFsize / R2, each (num_mesh_vertices,)

Output: data/finetuning/subject_cache/<subject>.<L|R>.pt (a dict of tensors).
Run from the repo root with the svt_sit env.
"""
import glob
import os
import os.path as osp
import re
import sys

import nibabel as nib
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.config import ENCODER_KWARGS
from models.patching import load_patch_indices, patchify

METRICS_ROOT = os.environ.get("FT_METRICS_ROOT", "data/hcp_ico6/metrics")
GT_ROOT = os.environ.get("FT_GT_ROOT", "data/hcp_ico6/gt")
OUT_ROOT = os.environ.get("FT_CACHE_ROOT", "data/finetuning/subject_cache")
INPUT_CHANNELS = ["curv", "sulc", "thickness"]        # must match pretraining
LABELS = ["polarAngle", "eccentricity", "pRFsize", "R2"]
ICO_GRID = int(os.environ.get("FT_ICO_GRID", ENCODER_KWARGS["ico_grid"]))


def _by_name(path, names):
    gii = nib.load(path)
    lut = {d.meta.get("Name"): d.data for d in gii.darrays}
    missing = [n for n in names if n not in lut]
    if missing:
        raise ValueError(f"{path} missing darray(s) {missing}, has {list(lut)}")
    return lut


def main():
    gt_files = sorted(glob.glob(osp.join(GT_ROOT, "*.gii")))
    if not gt_files:
        raise FileNotFoundError(f"no gt gii under {GT_ROOT}")
    os.makedirs(OUT_ROOT, exist_ok=True)
    patch_indices = load_patch_indices(ico_mesh=6, ico_grid=ICO_GRID, reorder=False)

    n_written, n_skipped = 0, 0
    for gt_path in gt_files:
        m = re.match(r"(sub-\d+)\.([LR])\.", osp.basename(gt_path))
        subject, hemi = m.group(1), m.group(2)
        metrics_path = osp.join(METRICS_ROOT, f"{subject}.{hemi}.ico6_fs_LR.shape.gii")
        if not osp.exists(metrics_path):
            n_skipped += 1
            continue

        feats = _by_name(metrics_path, INPUT_CHANNELS)
        x = torch.stack([torch.from_numpy(feats[c]).float() for c in INPUT_CHANNELS])  # (C, Vmesh)
        x = torch.nan_to_num(patchify(x, patch_indices), nan=0.0)  # (C, N, V)

        # Labels are NaN outside the fitted region (~non-visual vertices). Zero them:
        # R2=0 there means the R2-weighted loss and the R2>threshold eval mask both
        # exclude those vertices, and NaN never propagates into the loss.
        gt = _by_name(gt_path, LABELS)
        sample = {"x": x}
        for lab in LABELS:
            sample[lab] = torch.nan_to_num(torch.from_numpy(gt[lab]).float(), nan=0.0)  # (Vmesh,)

        torch.save(sample, osp.join(OUT_ROOT, f"{subject}.{hemi}.pt"))
        n_written += 1

    print(f"wrote {n_written} subject-hemisphere caches to {OUT_ROOT} "
          f"(x shape {tuple(x.shape)}), skipped {n_skipped} (no metrics)")


if __name__ == "__main__":
    main()
