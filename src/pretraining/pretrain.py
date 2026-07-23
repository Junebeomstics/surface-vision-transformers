"""Self-supervised pretraining of the SiT encoder on HCP surfaces.

Masked patch prediction (see mpp.py): the resulting checkpoint contains
the full LightningModule (encoder + decoder + optimizer state), from
which `encoder.*` weights can be loaded into SiT for downstream
fine-tuning (see src/finetuning).
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from dataset import HCPDataModule
from mpp import SiTPretraining
from models.config import ENCODER_KWARGS

# Each run is scoped to its own experiment folder so checkpoints/logs never
# collide across runs: experiments/<EXPERIMENT>/{checkpoints,logs}. Override
# EXPERIMENT / DATA_ROOT / epochs / batch per run via env vars (see CLAUDE.md).
EXPERIMENT = os.environ.get("PRETRAIN_EXPERIMENT", "default")
DATA_ROOT = os.environ.get("PRETRAIN_DATA_ROOT", "data/pretraining")
EXP_DIR = os.path.join("experiments", EXPERIMENT)
CKPT_DIR = os.path.join(EXP_DIR, "checkpoints")
LOG_DIR = os.path.join(EXP_DIR, "logs")
BATCH_SIZE = int(os.environ.get("PRETRAIN_BATCH_SIZE", 16))
NUM_WORKERS = int(os.environ.get("PRETRAIN_NUM_WORKERS", 8))
MAX_EPOCHS = int(os.environ.get("PRETRAIN_MAX_EPOCHS", 2))
LR = 1e-4
WEIGHT_DECAY = 0.05
MASK_RATIO = 0.6

# ENCODER_KWARGS imported from models.config -- shared with fine-tuning so the
# encoders (and thus the transferred pretrained weights) can never drift.


def main():
    datamodule = HCPDataModule(root=DATA_ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    model = SiTPretraining(ENCODER_KWARGS, mask_ratio=MASK_RATIO, lr=LR, weight_decay=WEIGHT_DECAY)

    # Runs on 1 gpu by default
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=CSVLogger(LOG_DIR, name="pretraining"),
        callbacks=[ModelCheckpoint(dirpath=CKPT_DIR, filename="pretrained-{epoch}-{val_loss:.4f}",
                                    monitor="val_loss", save_top_k=1, save_last=True)],
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)

    # full LightningModule checkpoint (encoder + decoder + optimizer state),
    # on top of the best/last checkpoints ModelCheckpoint already tracked above
    trainer.save_checkpoint(f"{CKPT_DIR}/pretrained.ckpt")


if __name__ == "__main__":
    main()
