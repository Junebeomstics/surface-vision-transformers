"""Wraps the ROI-only Retinotopy dataset into full-hemisphere SiT inputs.

`Retinotopy` (finetuning_dataset.py) gives per-vertex features/labels only
for the ROI (early visual cortex): variable-length, no patch/grid structure.
SiTDense expects patch-gridded input over the whole hemisphere and produces
predictions over the whole hemisphere too. So each subject's ROI data is
first scattered into a zero-filled full-mesh array, then the *input*
features are patchified onto the icosphere grid for the encoder; labels and
R2 are left in full-mesh-vertex space (zero outside the ROI) since the
model's output is already unpatched back to mesh vertices by SiTDense --
the R2-weighted loss then naturally restricts supervision to the ROI, same
as the original per-vertex model did.

This requires a static mapping from ROI row -> full ico6-mesh vertex index,
the same for every subject (RetinoSolver resamples everyone onto the same
template mesh, so the ROI occupies the same vertex positions for all of
them): data/finetuning/roi_vertex_index_{hemisphere}.pt, a LongTensor of
shape (num_roi_vertices,). This repo doesn't produce that file -- it has to
come from whatever pipeline built the ROI .pt files in the first place.
"""
import os.path as osp

import torch
from torch.utils.data import Dataset

from finetuning_dataset import Retinotopy
from models.patching import load_patch_indices, patchify


class SiTRetinotopy(Dataset):
    def __init__(self, root, split="Train", hemisphere="Left", prediction="polarAngle",
                 feature_bundle="myelincurv", seed=None, num_channels=3,
                 ico_mesh=6, ico_grid=4, reorder=False):
        # Load fine-tuning data (vertices level, visual cortex only)
        # So a roi is a set of vertices with polar angle and r2
        self.roi = Retinotopy(root, split=split, hemisphere=hemisphere, prediction=prediction,
                               feature_bundle=feature_bundle, seed=seed)

        actual_channels = self.roi[0].x.shape[1]
        if actual_channels != num_channels:
            raise ValueError(
                f"feature_bundle='{feature_bundle}' has {actual_channels} channel(s), but "
                f"num_channels={num_channels} was requested (must match the pretrained "
                "encoder's ENCODER_KWARGS['num_channels']). Regenerate the ROI .pt files with "
                "a feature bundle that has the same channels the encoder was pretrained on."
            )

        # Load mapping between ROI (i.e. surface vision areas) and its vertices
        # TODO: isn't this a subset of the general patch-vertices mapping
        index_path = osp.join(root, f"roi_vertex_index_{hemisphere}.pt")
        if not osp.exists(index_path):
            raise FileNotFoundError(
                f"{index_path} not found. SiTRetinotopy needs a static LongTensor mapping "
                "each ROI row to its full ico6-mesh vertex index (same for every subject) -- "
                "save one there from whatever pipeline produced the ROI .pt files."
            )
        self.roi_vertex_index = torch.load(index_path, weights_only=True)

        self.patch_indices = load_patch_indices(ico_mesh=ico_mesh, ico_grid=ico_grid, reorder=reorder)
        self.num_mesh_vertices = int(self.patch_indices.max().item()) + 1

    def __len__(self):
        return len(self.roi)

    def __getitem__(self, idx):
        subject = self.roi[idx]
        num_mesh = self.num_mesh_vertices

        # Compute vertices representation: those outside ROI are represented with a zero
        # TODO: do we want to replace this with R^2 weighting???
        x_full = torch.zeros(num_mesh, subject.x.shape[1], dtype=torch.float32)
        x_full[self.roi_vertex_index] = subject.x.float()
        y_full = torch.zeros(num_mesh, dtype=torch.float32)
        y_full[self.roi_vertex_index] = subject.y.view(-1).float()
        r2_full = torch.zeros(num_mesh, dtype=torch.float32)
        r2_full[self.roi_vertex_index] = subject.R2.view(-1).float()

        x_patches = patchify(x_full.T, self.patch_indices)  # C, N, V
        return x_patches, y_full, r2_full
