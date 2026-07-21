"""Self-supervised pretraining of the MS-SiT encoder on HCP surfaces.

Masked patch prediction (see mpp.py): the resulting checkpoint contains
the full LightningModule (encoder + decoder + optimizer state), from
which `encoder.*` weights can be loaded into MS-SiT for downstream
fine-tuning (see src/finetuning).
"""
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from dataset import HCPDataModule
from mpp import MSSiTPretraining

DATA_ROOT = "data/pretraining"
CKPT_DIR = "checkpoints"
BATCH_SIZE = 16
NUM_WORKERS = 8
MAX_EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 0.05
MASK_RATIO = 0.6

# MS-SiT tiny, ico_grid=4 (5120 patches x 15 vertices) -- see model.py / ICO_GRID
ENCODER_KWARGS = dict(
    ico_grid=4,
    num_channels=4,
    embed_dim=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    window_size=(64, 64, 64, 80),
    drop_path_rate=0.1,
)


def main():
    datamodule = HCPDataModule(root=DATA_ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    model = MSSiTPretraining(ENCODER_KWARGS, mask_ratio=MASK_RATIO, lr=LR, weight_decay=WEIGHT_DECAY)

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
