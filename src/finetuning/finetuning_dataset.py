"""
    Minimal reader for RetinoSolver's pre-processed ROI data.
"""
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset

_SPLIT = {"Train": "training", "Development": "development", "Test": "test"}
_PRED = {"polarAngle": "PA", "eccentricity": "ecc", "pRFsize": "pRFsize"}
_HEMI = {"Left": "LH", "Right": "RH"}


class Retinotopy(InMemoryDataset):
    def __init__(self, root, split="Train", hemisphere="Left",
                 prediction="polarAngle", feature_bundle="myelincurv", seed=None):
        self.hemisphere = hemisphere
        self.prediction = prediction
        self.feature_bundle = feature_bundle
        self.seed = seed
        super().__init__(root)
        path = self.processed_paths[list(_SPLIT).index(split)]
        # Load data + labels
        # data contains data from all subjects
        # slices contains single subject's indices (so we can retrieve single-subject data)
        # Structure of self.data:
        # - self.data.x: vertices features
        # - self.data.y: target label to be predicted for the vertex
        # - self.data.R2: variance explained by the vertex
        self.data, self.slices = torch.load(path, weights_only=False)

    @property
    def processed_dir(self):
        base = osp.join(self.root, "processed")
        return osp.join(base, f"seed{self.seed}") if self.seed is not None else base

    @property
    def processed_file_names(self):
        h, p = _HEMI[self.hemisphere], _PRED[self.prediction]
        b = f"_{self.feature_bundle}" if self.feature_bundle else ""
        return [f"{_SPLIT[s]}_{p}_{h}{b}_ROI.pt" for s in _SPLIT]

    # Files come from the upstream pipeline; skip PyG's download/process steps.
    def _download(self):
        pass

    def _process(self):
        pass