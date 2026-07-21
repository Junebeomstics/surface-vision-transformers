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

DATA_ROOT = "data/pretraining"
CKPT_DIR = "checkpoints"
BATCH_SIZE = 16
NUM_WORKERS = 8
MAX_EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 0.05
MASK_RATIO = 0.6

# SiT tiny, ico_grid=4 (5120 patches x 15 vertices) -- see model.py / ICO_GRID
# num_channels is set by whatever cortical features are stacked in the .pt files
# (e.g. 3 for curvature, thickness, sulcal depth) -- change freely, everything
# downstream (patch embedding, mask token, decoder) derives its shape from it.
ENCODER_KWARGS = dict(
    ico_grid=4,
    num_channels=3,
    embed_dim=192,
    depth=12,
    num_heads=3,
    dim_head=64,
    mlp_ratio=4,
)


def main():
    datamodule = HCPDataModule(root=DATA_ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    model = SiTPretraining(ENCODER_KWARGS, mask_ratio=MASK_RATIO, lr=LR, weight_decay=WEIGHT_DECAY)

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=CSVLogger("logs", name="pretraining"),
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
