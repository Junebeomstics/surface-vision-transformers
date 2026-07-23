"""Fine-tune the SiT encoder on HCP retinotopy, with the deepRetinotopy eval policy.

Input features are the SAME curv/sulc/thickness the encoder was pretrained on
(see prepare_finetuning.py); the model (SiTDense) predicts one value per mesh
vertex over the whole hemisphere. The retinotopy ROI is applied as a mask on the
output: supervision and evaluation are restricted to ROI vertices, not the input.

Evaluation follows deepRetinotopy's canonical NSD policy (see CLAUDE.md):
  - polarAngle -> circular correlation (astropy), eccentricity -> Pearson,
    pRFsize -> Spearman  (see metrics.PRIMARY_METRIC)
  - mask: EVC ROI (v1v3fovea) & R2 > 10 (0-100 scale) & (ecc GT <= 12 for ecc)
  - aggregate per subject first, then average across subjects, then across seeds.

Run with vs. without pretrained weights by pointing FT_PRETRAINED_CKPT at a
checkpoint or setting it to "none" (random init) -- this is the pretrained-vs-
scratch ablation. Each run writes to its own experiment folder (see CLAUDE.md).
"""
import os
import os.path as osp
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from models.dense_head import SiTDense
from models.config import ENCODER_KWARGS
from models.patching import load_patch_indices
from metrics import masked_circular_loss, regression_metrics, CIRCULAR, PRIMARY_METRIC
from retinotopy_dataset import RetinotopyDataset, read_subjects

# --- paths / experiment (env-overridable, see CLAUDE.md) ---
CACHE_ROOT = os.environ.get("FT_CACHE_ROOT", "data/finetuning/subject_cache")
ROI_ROOT = os.environ.get("FT_ROI_ROOT", "data/finetuning")
SPLITS_ROOT = os.environ.get("FT_SPLITS_ROOT",
                             "/mnt/scratch/junb/deepRetinotopy/Retinotopy/data/subject_splits")
ROI = os.environ.get("FT_ROI", "v1v3fovea")          # EVC per eval policy
EXPERIMENT = os.environ.get("FT_EXPERIMENT", "finetune")
EXP_DIR = osp.join("experiments", EXPERIMENT)

# pretrained-vs-scratch ablation: a ckpt path, or "none" for random init
PRETRAINED_CKPT = os.environ.get("FT_PRETRAINED_CKPT",
                                 "experiments/hcp1112_ico2_sit/checkpoints/pretrained.ckpt")
USE_PRETRAINED = PRETRAINED_CKPT.strip().lower() not in ("", "none")

SEEDS = [int(s) for s in os.environ.get("FT_SEEDS", "0,1,2").split(",")]
PREDICTIONS = os.environ.get("FT_PREDICTIONS", "polarAngle").split(",")
HEMISPHERES = os.environ.get("FT_HEMISPHERES", "Left,Right").split(",")
BATCH_SIZE = int(os.environ.get("FT_BATCH_SIZE", 8))
MAX_EPOCHS = int(os.environ.get("FT_MAX_EPOCHS", 100))
PATIENCE = int(os.environ.get("FT_PATIENCE", 30))
LR = float(os.environ.get("FT_LR", 1e-5))

R2_THR = 10.0        # deepRetinotopy policy: R2 > 10 (0-100 scale)
ECC_GT_MAX = 12.0    # eccentricity extra mask: GT ecc <= 12
# hemisphere to apply deepRetinotopy's 180-deg polar-angle re-referencing to
# (its PA is centered near 180 and otherwise collapses to the antipode)
REREF_PA_HEMI = os.environ.get("FT_REREF_PA_HEMI", "Right")

_SPLIT_FILE = {"Train": "train_subjects.txt", "Development": "dev_subjects.txt",
               "Test": "test_subjects.txt"}
NUM_MESH_VERTICES = int(load_patch_indices(ico_mesh=6, ico_grid=ENCODER_KWARGS["ico_grid"],
                                           reorder=False).max()) + 1


def build_roi_mask(hemisphere, roi, num_vertices):
    """(num_vertices,) bool mask, True at ROI vertices (see roi_vertex_index_*.pt)."""
    idx = torch.load(osp.join(ROI_ROOT, f"roi_vertex_index_{hemisphere}_{roi}.pt"),
                     weights_only=True)
    mask = torch.zeros(num_vertices, dtype=torch.bool)
    mask[idx] = True
    return mask


class SiTFineTune(L.LightningModule):
    def __init__(self, encoder_kwargs, prediction, roi_mask, lr=LR,
                 r2_thr=R2_THR, ecc_gt_max=ECC_GT_MAX):
        super().__init__()
        self.save_hyperparameters(ignore=["roi_mask"])
        self.model = SiTDense(encoder_kwargs, num_classes=1)
        self.prediction = prediction
        self.r2_thr = r2_thr
        self.ecc_gt_max = ecc_gt_max
        self.register_buffer("roi_mask", roi_mask)   # (num_mesh_vertices,) bool
        self._test = None

    def load_pretrained_encoder(self, ckpt_path):
        # the pretraining checkpoint holds a different LightningModule (encoder +
        # masked-patch-prediction decoder, see pretraining/mpp.py) -- only the
        # "encoder.*" weights are structurally shared with this one.
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        encoder_state = {k[len("encoder."):]: v for k, v in state_dict.items()
                         if k.startswith("encoder.")}
        missing, unexpected = self.model.encoder.load_state_dict(encoder_state, strict=False)
        print(f"loaded pretrained encoder from {ckpt_path}: {len(encoder_state)} tensors "
              f"({len(missing)} missing, {len(unexpected)} unexpected)")

    def forward(self, x):
        return self.model(x)

    def _loss(self, pred, target, r2):
        # ROI-restricted, R2-weighted; circular for polarAngle (wraps at 0/360).
        m = self.roi_mask
        pred, target, r2 = pred[:, m], target[:, m], r2[:, m]
        if self.prediction in CIRCULAR:
            return masked_circular_loss(pred, target, r2)
        return F.smooth_l1_loss(r2 * pred, r2 * target)

    def training_step(self, batch, _):
        x, y, r2, _ = batch
        loss = self._loss(self(x), y, r2)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, _):
        x, y, r2, _ = batch
        loss = self._loss(self(x), y, r2)
        self.log("val_loss", loss, prog_bar=True, batch_size=x.shape[0])

    def on_test_epoch_start(self):
        self._test = []

    def test_step(self, batch, _):
        x, y, r2, ecc = batch
        pred = self(x)   # (B, num_mesh_vertices)
        # keep each subject separate -- correlations are computed per subject.
        for i in range(pred.shape[0]):
            self._test.append((pred[i].detach().cpu().numpy(), y[i].cpu().numpy(),
                               r2[i].cpu().numpy(), ecc[i].cpu().numpy()))

    def on_test_epoch_end(self):
        # deepRetinotopy policy: per-subject metric over EVC ROI & R2>10 (& ecc<=12),
        # then averaged across subjects. regression_metrics applies R2>r2_thr itself.
        roi = self.roi_mask.cpu().numpy()
        primary = PRIMARY_METRIC[self.prediction]
        per_subject = []
        for pred, y, r2, ecc in self._test:
            sel = roi.copy()
            if self.prediction == "eccentricity":
                sel = sel & (ecc <= self.ecc_gt_max)
            m = regression_metrics(pred[sel], y[sel], r2[sel], self.prediction, self.r2_thr)
            if primary in m and np.isfinite(m[primary]):
                per_subject.append(m[primary])
        score = float(np.mean(per_subject)) if per_subject else float("nan")
        self.log(f"test_{primary}", score)
        self._test = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def _dataset(split, seed, hemisphere, prediction):
    subjects = read_subjects(osp.join(SPLITS_ROOT, f"seed{seed}", _SPLIT_FILE[split]))
    return RetinotopyDataset(CACHE_ROOT, subjects, hemisphere=hemisphere, prediction=prediction,
                             reref_pa=(hemisphere == REREF_PA_HEMI))


def run_fold(seed, prediction, hemisphere):
    train_ds = _dataset("Train", seed, hemisphere, prediction)
    val_ds = _dataset("Development", seed, hemisphere, prediction)
    test_ds = _dataset("Test", seed, hemisphere, prediction)
    load = lambda ds, shuffle=False: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    roi_mask = build_roi_mask(hemisphere, ROI, NUM_MESH_VERTICES)
    model = SiTFineTune(ENCODER_KWARGS, prediction=prediction, roi_mask=roi_mask)
    if USE_PRETRAINED:
        model.load_pretrained_encoder(PRETRAINED_CKPT)   # else: random-init ablation

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=CSVLogger(osp.join(EXP_DIR, "logs"), name=f"{prediction}/{hemisphere}/seed{seed}"),
        callbacks=[ModelCheckpoint(dirpath=osp.join(EXP_DIR, "checkpoints",
                                                    f"{prediction}_{hemisphere}_seed{seed}"),
                                   monitor="val_loss", save_top_k=1),
                   EarlyStopping(monitor="val_loss", patience=PATIENCE)],
        log_every_n_steps=1,
    )
    trainer.fit(model, load(train_ds, shuffle=True), load(val_ds))
    # ckpt_path="best" -> evaluate the early-stopped (best-dev) weights, not the last.
    result = trainer.test(model, load(test_ds), ckpt_path="best")
    return result[0][f"test_{PRIMARY_METRIC[prediction]}"]


def main():
    tag = f"YES ({PRETRAINED_CKPT})" if USE_PRETRAINED else "NO (random init)"
    print(f"[fine-tune] experiment={EXPERIMENT} | pretrained={tag} | roi={ROI}")
    for prediction in PREDICTIONS:
        primary = PRIMARY_METRIC[prediction]
        for hemisphere in HEMISPHERES:
            scores = [run_fold(seed, prediction, hemisphere) for seed in SEEDS]
            print(f"\n[{prediction}/{hemisphere}] test {primary} over {len(SEEDS)} seeds: "
                  f"{np.nanmean(scores):.4f} +/- {np.nanstd(scores):.4f}  {scores}")


if __name__ == "__main__":
    main()
