"""Fine-tuning + test evaluation on the RetinoSolver ROI data.

Train on the train split, early-stop on the development split, and estimate the
generalization error on the test split. The cross-validation loop is in code: it
iterates over the per-seed splits (each `processed/seed{N}/` is one train/dev/
test partition), refits, and aggregates the test metric as mean +/- std -- no
need to relaunch the script per seed.
"""
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch_geometric.loader import DataLoader

from finetuning_dataset import Retinotopy

ROOT = "data/finetuning"
PRETRAINED_CKPT = "checkpoints/pretrained.ckpt"   # weights to fine-tune from
SEEDS = [0, 1, 2, 3, 4]   # each selects processed/seed{N}/ -> one CV fold
BATCH_SIZE = 8
MAX_EPOCHS = 100
PATIENCE = 30
# 
R2_THR = 2.2              # evaluate test error only on reliably-fit vertices


# Mock network: to be replaced with proper network
class MLP(L.LightningModule):
    def __init__(self, in_channels, hidden=128, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def training_step(self, batch, _):
        pred, target, r2 = self(batch.x), batch.y.view(-1), batch.R2.view(-1)
        loss = F.smooth_l1_loss(r2 * pred, r2 * target)   # R2-weighted, as upstream
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch.num_nodes)
        return loss

    def validation_step(self, batch, _):
        pred, target, r2 = self(batch.x), batch.y.view(-1), batch.R2.view(-1)
        loss = F.smooth_l1_loss(r2 * pred, r2 * target)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_nodes)

    def test_step(self, batch, _):
        pred, target, r2 = self(batch.x), batch.y.view(-1), batch.R2.view(-1)
        mask = r2 > R2_THR                                # reliable vertices only
        mae = (pred[mask] - target[mask]).abs().mean()    # generalization error
        self.log("test_mae", mae, batch_size=int(mask.sum()))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def run_fold(seed):
    train_ds = Retinotopy(ROOT, "Train", seed=seed)
    val_ds = Retinotopy(ROOT, "Development", seed=seed)
    test_ds = Retinotopy(ROOT, "Test", seed=seed)
    load = lambda ds, shuffle=False: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    model = MLP.load_from_checkpoint(PRETRAINED_CKPT)   # start from pretrained weights
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=CSVLogger("logs", name=f"seed{seed}"),   # -> logs/seed{N}/version_*/metrics.csv
        callbacks=[ModelCheckpoint(monitor="val_loss", save_top_k=1),
                   EarlyStopping(monitor="val_loss", patience=PATIENCE)],
        log_every_n_steps=1,
    )
    trainer.fit(model, load(train_ds, shuffle=True), load(val_ds))
    # ckpt_path="best" -> evaluate the early-stopped (best-dev) weights, not the last.
    result = trainer.test(model, load(test_ds), ckpt_path="best")
    return result[0]["test_mae"]


def main():
    scores = [run_fold(seed) for seed in SEEDS]
    print(f"\nTest MAE over {len(SEEDS)} folds: "
          f"{np.mean(scores):.4f} +/- {np.std(scores):.4f}")


if __name__ == "__main__":
    main()