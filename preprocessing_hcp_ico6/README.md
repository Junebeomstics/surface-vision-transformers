# HCP FreeSurfer → ico6 preprocessing for MS-SiT

Converts HCP FreeSurfer native surfaces into the ico6 (40,962-vertex) mesh the
surface-vision-transformers `metrics` dataloader expects, vertex-aligned so
inputs and ground truth share the same vertex indices.

- **Input features** (`convert_metrics_to_ico6.sh`): curv, sulc, thickness →
  `<subj>.<L|R>.ico6_fs_LR.shape.gii`, 3 darrays `[curv, sulc, thickness]`.
  Runs on every subject with the FreeSurfer binary surfaces (sphere.reg, curv,
  sulc, thickness, white, pial); the GIFTI sphere and midthickness are generated
  on the fly in a temp dir when missing, so no pre-conversion is required.
- **Ground truth** (`convert_gt_to_ico6.py`): polarAngle, eccentricity, pRFsize,
  R2 → `<subj>.<L|R>.ico6_fs_LR.gt.shape.gii`, 5 maps
  `[polarAngle, eccentricity, pRFsize, R2, valid]`. Only for subjects that have
  native retinotopy under `converted_native/` (181 HCP subjects).

RH is resampled onto the L ico6 template, giving left/right vertex
correspondence (the "symmetrisation" the dataloader assumes).

## Layout

```
preprocessing_hcp_ico6/
  convert_metrics_to_ico6.sh   input features → ico6 (bash + wb_command/FreeSurfer via container)
  convert_gt_to_ico6.py        retinotopy GT → ico6 (mask-aware, circular-safe)
  qc_ico6_metrics.py           full-cohort QC for the input files
  qc_ico6_gt.py                full-cohort QC for the GT files
  build_templates.sh           reproduce the resample-target templates (Phase 0)
  templates/                   ico6-deformed_to-fsaverage.{L,R}.surf.gii, atlasroi_ico6.shape.gii
  subjects_all.txt             all 1112 FreeSurfer subjects
  subjects_gt181.txt           181 subjects with retinotopy GT
```

## Dependencies

- Connectome Workbench + FreeSurfer, provided by the extracted container sandbox
  `SANDBOX` (default `/mnt/scratch/junb/deepRetinotopy/.sif_sandbox`, from
  `deepretinotopy_1.0.18.sif`).
- fs_LR standard-mesh atlases (only to rebuild templates): `ATLASES`.

All roots are env-overridable: `SUBJECTS_ROOT`, `SANDBOX`, `TEMPLATES`,
`OUT_DIR`, and for GT `GT_ROOT`, `DR_REPO`.

## Usage

```bash
# input features, all subjects, 12-way parallel
OUT_DIR=../data/hcp_ico6/metrics ./convert_metrics_to_ico6.sh -l subjects_all.txt -j 12

# ground truth, the 181 with retinotopy
python convert_gt_to_ico6.py -l subjects_gt181.txt -o ../data/hcp_ico6/gt -j 8

# QC
python qc_ico6_metrics.py -m ../data/hcp_ico6/metrics -o ../data/hcp_ico6/qc_metrics.csv
python qc_ico6_gt.py      -g ../data/hcp_ico6/gt      -o ../data/hcp_ico6/qc_gt.csv
```

Per-subject wall time ≈ 9 s (both hemispheres). GPU is not used.

## Method notes

- **Single-interpolation resample.** `templates/ico6-deformed_to-fsaverage.<H>.surf.gii`
  carries the ico6 mesh in fsaverage coordinates, so one ADAP_BARY_AREA resample
  replaces native → fs_LR → ico6. Cross-validated vs the two-step path at r=0.9991.
- **L/R correspondence.** The R template projects the ico6-L mesh through the
  **L** fs_LR sphere (not R), because fs_LR L/R spheres are exact x-mirrors;
  using R would flip left/right. Verified: sulc L/R vertexwise corr ≈ 0.78.
- **GT is mask-aware and circular-safe.** Invalid GT vertices are NaN and must
  not bleed during interpolation; polarAngle wraps at 0/360. Both are handled by
  resampling cos/sin(polarAngle) and a validity channel in a NaN-free linear
  space, then dividing by the resampled validity and atan2-ing back. Off-mask
  output is NaN; R2 (0–100 scale) is stored so training can mask on `R2 > 10`.
- **Mask.** `templates/atlasroi_ico6.shape.gii` is the fs_LR cortex ROI on ico6
  (Dice 0.98 vs the SVT-shipped dHCP medial-wall mask, confirming shared frame).

## Scope

Input features: 1112 subjects. GT: 181 (retinotopy-available upper bound). The
other 931 have inputs only — usable for masked-surface-modelling pretraining,
not supervised retinotopy.
