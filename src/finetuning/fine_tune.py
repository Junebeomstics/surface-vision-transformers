"""Fine-tuning + test evaluation on the RetinoSolver ROI data, using SiT.

Train on the train split, early-stop on the development split, and estimate the
generalization error on the test split. The cross-validation loop is in code: it
iterates over the per-seed splits (each `processed/seed{N}/` is one train/dev/
test partition), refits, and aggregates the test metric as mean +/- std -- no
need to relaunch the script per seed.

SiT here is the same encoder pretrained in src/pretraining, plus a per-token
linear head (SiTDense, see src/models/dense_head.py) that predicts one value
per mesh vertex across the whole hemisphere. Loss/eval stay R2-weighted and
ROI-restricted exactly as before -- R2 is zero outside the ROI (see
sit_dataset.py), which zeroes the loss there for free.
"""
import os
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
from sit_dataset import SiTRetinotopy

ROOT = "data/finetuning"
PRETRAINED_CKPT = "checkpoints/pretrained.ckpt"   # weights to fine-tune from
SEEDS = [0, 1, 2, 3, 4]   # each selects processed/seed{N}/ -> one CV fold
PREDICTIONS = ["polarAngle"]   # one independent model per quantity
HEMISPHERES = ["Left", "Right"]   # one independent model per hemisphere
BATCH_SIZE = 8
MAX_EPOCHS = 100
PATIENCE = 30
R2_THR = 2.2              # evaluate test error only on reliably-fit vertices

# must match the ENCODER_KWARGS the pretrained checkpoint (PRETRAINED_CKPT) used
ENCODER_KWARGS = dict(
    ico_grid=2,
    num_channels=3,
    embed_dim=192,
    depth=12,
    num_heads=3,
    dim_head=64,
    mlp_ratio=4,
)


class SiTFineTune(L.LightningModule):
    def __init__(self, encoder_kwargs, lr=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = SiTDense(encoder_kwargs, num_classes=1)

    def load_pretrained_encoder(self, ckpt_path):
        # the pretraining checkpoint holds a different LightningModule (encoder +
        # masked-patch-prediction decoder, see pretraining/mpp.py) -- only the
        # "encoder.*" weights are structurally shared with this one.
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        encoder_state = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}
        missing, unexpected = self.model.encoder.load_state_dict(encoder_state, strict=False)
        print(f"loaded pretrained encoder: {len(encoder_state)} tensors "
              f"({len(missing)} missing, {len(unexpected)} unexpected)")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, target, r2 = batch
        pred = self(x)
        loss = F.smooth_l1_loss(r2 * pred, r2 * target)   # R2-weighted, as upstream
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, _):
        x, target, r2 = batch
        pred = self(x)
        loss = F.smooth_l1_loss(r2 * pred, r2 * target)
        self.log("val_loss", loss, prog_bar=True, batch_size=x.shape[0])

    def test_step(self, batch, _):
        x, target, r2 = batch
        pred = self(x)
        mask = r2 > R2_THR                                # reliable vertices only
        mae = (pred[mask] - target[mask]).abs().mean()    # generalization error
        self.log("test_mae", mae, batch_size=int(mask.sum()))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def run_fold(seed, prediction, hemisphere):
    common = dict(seed=seed, prediction=prediction, hemisphere=hemisphere,
                  num_channels=ENCODER_KWARGS["num_channels"])
    train_ds = SiTRetinotopy(ROOT, "Train", **common)
    val_ds = SiTRetinotopy(ROOT, "Development", **common)
    test_ds = SiTRetinotopy(ROOT, "Test", **common)
    load = lambda ds, shuffle=False: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    model = SiTFineTune(ENCODER_KWARGS)
    model.load_pretrained_encoder(PRETRAINED_CKPT)   # start from pretrained weights

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        # -> logs/{prediction}/{hemisphere}/seed{N}/version_*/metrics.csv
        logger=CSVLogger("logs", name=f"{prediction}/{hemisphere}/seed{seed}"),
        callbacks=[ModelCheckpoint(monitor="val_loss", save_top_k=1),
                   EarlyStopping(monitor="val_loss", patience=PATIENCE)],
        log_every_n_steps=1,
    )
    trainer.fit(model, load(train_ds, shuffle=True), load(val_ds))
    # ckpt_path="best" -> evaluate the early-stopped (best-dev) weights, not the last.
    result = trainer.test(model, load(test_ds), ckpt_path="best")
    return result[0]["test_mae"]


def main():
    for prediction in PREDICTIONS:
        for hemisphere in HEMISPHERES:
            scores = [run_fold(seed, prediction, hemisphere) for seed in SEEDS]
            print(f"\n[{prediction}/{hemisphere}] test MAE over {len(SEEDS)} folds: "
                  f"{np.mean(scores):.4f} +/- {np.std(scores):.4f}")


if __name__ == "__main__":
    main()
