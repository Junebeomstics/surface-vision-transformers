"""
Masked surface modelling pretraining for MS-SiT.

The existing tools/pretrain.py only builds SiT (`if config['transformer']['model']
== 'SiT'`, with no other branch), and models/mpp.py depends on SiT-only
attributes. This script is the MS-SiT counterpart: it wraps an MS-SiT U-Net in
models/ms_mae.masked_surface_modelling and trains it to reconstruct masked
cortical patches.

Pretraining needs no labels, so this does not use the labelled `metrics`
dataloader. It reads merged .shape.gii files straight from a directory.

Usage:
    cd tools
    python pretrain_ms_sit.py ../config/MS-SiT/pretraining/hparams.yml
"""

import argparse
import os
import sys
import glob
import math
import yaml
from datetime import datetime

import numpy as np
import nibabel as nb
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ms_sit_unet import MSSiTUNet
from models.ms_mae import masked_surface_modelling, patch_to_vertex_index


class CorticalMetricsUnlabelled(Dataset):
    """Unlabelled ico6 surface metrics, patched for MS-SiT.

    Returns (C, num_patches, num_vertices) float tensors.
    """

    def __init__(self, files, patch_to_vertex, channels, mask=None,
                 normalise="sub-standardise", clipping=True, clip_bounds=None):
        self.files = files
        self.p2v = patch_to_vertex.numpy()
        self.channels = channels
        self.mask = mask
        self.normalise = normalise
        self.clipping = clipping
        self.clip_bounds = clip_bounds

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.asarray(nb.load(self.files[idx]).agg_data(), dtype=np.float32)
        if data.ndim == 1:
            data = data[None, :]
        data = data[self.channels, :]

        if self.clipping and self.clip_bounds is not None:
            for i, (lo, hi) in enumerate(self.clip_bounds):
                data[i] = np.clip(data[i], lo, hi)

        if self.normalise == "sub-standardise":
            # standardise over cortex only, so the medial wall cannot shift the
            # mean of every channel
            sel = self.mask if self.mask is not None else slice(None)
            mu = data[:, sel].mean(axis=1, keepdims=True)
            sd = data[:, sel].std(axis=1, keepdims=True)
            data = (data - mu) / np.clip(sd, 1e-6, None)

        patches = data[:, self.p2v]           # C, num_patches, num_vertices
        return torch.from_numpy(np.ascontiguousarray(patches))


def build_loaders(config, p2v):
    d = config["data"]
    if d.get("file_list"):
        with open(d["file_list"]) as fh:
            files = [ln.strip() for ln in fh if ln.strip()]
    else:
        files = sorted(glob.glob(os.path.join(d["path_to_metrics"], "*.shape.gii")))
    if not files:
        raise RuntimeError(f"no .shape.gii files found under {d['path_to_metrics']}")

    mask = None
    if d.get("path_to_mask"):
        mask = np.asarray(nb.load(d["path_to_mask"]).agg_data()).ravel() > 0.5

    rng = np.random.RandomState(config["training"]["seed"])
    order = rng.permutation(len(files))
    n_val = max(1, int(len(files) * d["val_fraction"]))
    val_files = [files[i] for i in order[:n_val]]
    train_files = [files[i] for i in order[n_val:]]

    print(f"train files: {len(train_files)} | val files: {len(val_files)}")

    common = dict(patch_to_vertex=p2v, channels=d["channels"], mask=mask,
                  normalise=d["normalise"], clipping=d["clipping"],
                  clip_bounds=d.get("clip_bounds"))
    train_ds = CorticalMetricsUnlabelled(train_files, **common)
    val_ds = CorticalMetricsUnlabelled(val_files, **common)

    t = config["training"]
    return (
        DataLoader(train_ds, batch_size=t["bs"], shuffle=True,
                   num_workers=t["num_workers"], pin_memory=True, drop_last=True),
        DataLoader(val_ds, batch_size=t["bs_val"], shuffle=False,
                   num_workers=t["num_workers"], pin_memory=True),
    )


def build_model(config, device):
    mr, tr = config["mesh_resolution"], config["transformer"]
    n_channels = len(config["data"]["channels"])

    net = MSSiTUNet(
        ico_init_resolution=mr["ico_grid"],
        num_channels=n_channels,
        num_classes=n_channels,          # reconstruction, not classification
        embed_dim=tr["dim"],
        depths=tr["depth"],
        num_heads=tr["heads"],
        window_size=tr["window_size"],
        window_size_factor=tr["window_size_factor"],
        mlp_ratio=tr["mlp_ratio"],
        dropout=tr["dropout"],
        attention_dropout=tr["attention_dropout"],
        dropout_path=tr["drop_path_rate"],
        use_pos_emb=tr["use_pos_emb"],
        reorder=mr["reorder"],
        device=config["training"]["gpu"],
        path_to_workdir=config["data"]["path_to_workdir"],
    ).to(device)
    return net


def lr_at(epoch, config):
    """Linear warmup then cosine decay."""
    t = config["training"]
    warm, total, base = t["warmup_epochs"], t["epochs"], t["LR"]
    if epoch < warm:
        return base * (epoch + 1) / warm
    prog = (epoch - warm) / max(1, total - warm)
    return base * 0.5 * (1 + math.cos(math.pi * prog))


def main():
    ap = argparse.ArgumentParser(description="MS-SiT masked surface modelling")
    ap.add_argument("config", help="path to hparams.yml")
    args = ap.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    if config["MODEL"] != "ms-sit":
        raise ValueError(f"this script is MS-SiT only, got MODEL={config['MODEL']}")

    t = config["training"]
    torch.manual_seed(t["seed"])
    np.random.seed(t["seed"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; refusing to pretrain on CPU")
    device = torch.device(f"cuda:{t['gpu']}")
    print(f"device: {device} ({torch.cuda.get_device_name(t['gpu'])})")

    net = build_model(config, device)
    p2v = patch_to_vertex_index(net.indices, net.num_patches)

    msm = config["pretraining_msm"]
    ssl = masked_surface_modelling(
        transformer=net,
        patch_to_vertex=p2v,
        num_channels=len(config["data"]["channels"]),
        mask_prob=msm["mask_prob"],
        replace_prob=msm["replace_prob"],
        norm_target=msm["norm_target"],
        loss_mask_mode=msm.get("loss_mask_mode", "intersection"),
    ).to(device)
    print(f"loss mask mode: {ssl.loss_mask_mode}")

    n_par = sum(p.numel() for p in ssl.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_par:,}")

    train_loader, val_loader = build_loaders(config, p2v)

    opt = optim.AdamW(ssl.parameters(), lr=t["LR"],
                      weight_decay=config["AdamW"]["weight_decay"])

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(config["logging"]["folder_to_save_model"], stamp)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "hparams.yml"), "w") as fh:
        yaml.dump(config, fh)
    print(f"logging to {outdir}")

    best_val = float("inf")
    for epoch in range(t["epochs"]):
        lr = lr_at(epoch, config)
        for g in opt.param_groups:
            g["lr"] = lr

        ssl.train()
        running, nb_it = 0.0, 0
        for inputs in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            loss, stats = ssl(inputs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            nb_it += 1
        train_loss = running / max(1, nb_it)

        msg = f"epoch {epoch+1}/{t['epochs']}  lr {lr:.2e}  train {train_loss:.4f}"

        if (epoch + 1) % t["val_epoch"] == 0:
            ssl.eval()
            vrun, vit = 0.0, 0
            with torch.no_grad():
                for inputs in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    vloss, _ = ssl(inputs)
                    vrun += vloss.item()
                    vit += 1
            val_loss = vrun / max(1, vit)
            msg += f"  val {val_loss:.4f}"

            if t["save_ckpt"] and val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {"epoch": epoch,
                     "model_state_dict": net.state_dict(),
                     "ssl_state_dict": ssl.state_dict(),
                     "optimizer_state_dict": opt.state_dict(),
                     "val_loss": val_loss},
                    os.path.join(outdir, "encoder-best.pt"),
                )
                msg += "  [saved]"

        print(msg, flush=True)

    print(f"done. best val loss {best_val:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
