# surface-vision-transformers — project conventions

SiT (Surface Vision Transformer) **self-supervised pretraining** on HCP cortical
surfaces, then **retinotopy fine-tuning**. Code lives under `src/` (models /
pretraining / finetuning). This file defines the conventions to follow when
running experiments here.

## Goal & pipeline overview

Predict retinotopic maps (polarAngle, eccentricity, pRFsize) from HCP cortical
surface data. Only 181 of the 1112 available HCP subjects have retinotopy
ground truth (early visual cortex ROI only), so the strategy is: self-supervised
masked-patch-prediction (MPP) pretraining of the SiT encoder on all ~1112
subjects (unlabeled, curv/sulc/thickness features), then transfer the encoder
weights into supervised fine-tuning restricted to the 181 labeled subjects.
Fine-tuning trains one model per target (polarAngle/eccentricity/pRFsize) per
hemisphere, with 5-fold CV over seeded splits. polarAngle uses a circular
loss/metric (wraps at 0/360°); eccentricity/pRFsize use Smooth-L1 +
Pearson/Spearman. Evaluation is always restricted to vertices with R2 above a
threshold.

The repo has **two parallel implementations**: a legacy yaml-config-driven
pipeline (`tools/` + `models/` + `config/MS-SiT/`, MS-SiT U-Net backbone) and
the newer code-driven pipeline (`src/pretraining/`, `src/finetuning/`,
`src/models/`, SiT-tiny encoder) that this file's conventions target. Active
development is on `src/`.

## Environment

Use the **`svt_sit`** conda env for everything (torch 2.10+cu128, supports the
Blackwell GPU; full stack: lightning, timm, torch_geometric, astropy, nibabel):

```
conda activate svt_sit
# or call directly: /home/junb/miniconda3/envs/svt_sit/bin/python
```

The base env's torch is CUDA-11.8 and **cannot** use GPU 1 (RTX PRO 6000
Blackwell, sm_120) — always use `svt_sit`.

## Experiment folders (checkpoint management) — REQUIRED

**Every training run writes to its own experiment folder.** Never write
checkpoints/logs to shared top-level dirs. Layout:

```
experiments/<experiment-name>/
    checkpoints/    # ModelCheckpoint best/last + pretrained.ckpt
    logs/           # CSVLogger metrics.csv + hparams.yaml
```

`experiments/` is gitignored. Name experiments descriptively, encoding the data
and key hyperparameters, e.g. `hcp1112_ico2_sit`, `hcp181_ico2_2ch`.

`src/pretraining/pretrain.py` is driven entirely by env vars — no code edits per
run:

| env var | default | meaning |
|---|---|---|
| `PRETRAIN_EXPERIMENT` | `default` | experiment folder name → `experiments/<name>/` |
| `PRETRAIN_DATA_ROOT` | `data/pretraining` | dir holding `train/*.pt` and `val/*.pt` |
| `PRETRAIN_MAX_EPOCHS` | `2` | 2 is a smoke default; set higher for real runs |
| `PRETRAIN_BATCH_SIZE` | `16` | fine at ico_grid=2 (320 patches) |

Example (full-cohort pretraining):

```
cd <repo root>
CUDA_VISIBLE_DEVICES=0 \
PRETRAIN_EXPERIMENT=hcp1112_ico2_sit \
PRETRAIN_DATA_ROOT=data/hcp_ico6/pretraining_pt_ico2 \
PRETRAIN_MAX_EPOCHS=100 \
/home/junb/miniconda3/envs/svt_sit/bin/python src/pretraining/pretrain.py
```

Always run from the repo root (paths are relative to CWD). Pin a single GPU with
`CUDA_VISIBLE_DEVICES` to avoid mixed-arch DDP across the two different GPUs.

## Encoder config — single source of truth

`src/models/config.py` holds `ENCODER_KWARGS`. Both pretraining and fine-tuning
import it, so the encoder they build (and thus the transferred `encoder.*`
weights) can never drift. **Do not** redefine `ENCODER_KWARGS` locally in a
script. Current setting: `ico_grid=2` (320 patches × 153 vertices), 3 channels
(curv/sulc/thickness).

The **patch grid must match the encoder**: `prepare_data.py` reads `ico_grid`
from the same config by default. If you change `ico_grid`, re-patchify the data.

## Data layout

```
data/hcp_ico6/metrics/            # 1112 subj × 2 hemi ico6 gii [curv,sulc,thickness]
data/hcp_ico6/gt/                 # 181 subj retinotopy GT [PA,ecc,pRFsize,R2,valid]
data/hcp_ico6/pretraining_pt_ico2/{train,val}/   # patchified (3,320,153) pt
```

Patchify raw ico6 gii → pretraining `.pt`:

```
PREPARE_RAW_ROOT=data/hcp_ico6/metrics \
PREPARE_OUT_ROOT=data/hcp_ico6/pretraining_pt_ico2 \
PREPARE_VAL_FRACTION=0.05 \
/home/junb/miniconda3/envs/svt_sit/bin/python src/pretraining/prepare_data.py
```

Pretraining input features = fine-tuning input features (**curv/sulc/thickness**,
the same 3 the encoder was pretrained on). The retinotopy ROI is applied as a
**loss mask on the output** (which vertices count), not as an input restriction.

**Leakage note:** the 181 retinotopy (GT) subjects double as fine-tuning
train/dev/test. Pretraining on the full 1112 (incl. those 181) lets fine-tuning
test subjects be seen — unsupervised — during pretraining. For strict eval,
pretrain only on the 931 metrics-only subjects (no GT).

## Evaluation criteria (retinotopy) — official policy

Mirror deepRetinotopy's canonical NSD evaluation so our SiT numbers are
comparable (source: `/mnt/scratch/junb/deepRetinotopy/CLAUDE.md` lines 62-87 +
`nsd_evaluation/`). **Primary metric per target:**

| target | primary metric | how | notes |
|---|---|---|---|
| polarAngle | **circular correlation** (astropy `circcorrcoef`) | `circcorrcoef(deg2rad(mod(pred,360)), deg2rad(mod(gt,360)))` | degrees→mod360→radians. **Never score PA with Pearson alone.** |
| eccentricity | **Pearson** (+ Spearman) | `pearsonr` on linear degrees | extra mask: GT ecc ≤ 12 |
| pRFsize | **Spearman** (+ Pearson) | `spearmanr` | linear |

**Evaluation mask** (all must hold, per vertex):
`isfinite(pred) & isfinite(gt) & gt != 0 & (R2 > 10) & ROI ∈ EVC`
- **R2 > 10**, strict `>`. R2 here is **0–100 scale** (10 = 10% explained
  variance); `data/hcp_ico6/gt` R2 is on this scale (range ~0–77). If a source
  stores R2 as 0–1, use `> 0.1` instead.
- **EVC ROI** = V1/V2/V3 (ventral+dorsal), ROI label ids `{1,2,3,4,5,6}`. The
  `data/finetuning/roi_vertex_index_*_v1v3fovea.pt` map corresponds to this.
- eccentricity additionally masks GT ecc ≤ 12.

**Aggregation:** compute the metric **per subject first, then average across
subjects** (per seed); then average across seeds 0/1/2. Do **not** pool all
vertices from all subjects into one correlation.

## Git

Trunk is `federico/fine_tuning`. Feature branches for new work; keep the
`ENCODER_KWARGS` single-source and experiment-folder conventions above intact.
