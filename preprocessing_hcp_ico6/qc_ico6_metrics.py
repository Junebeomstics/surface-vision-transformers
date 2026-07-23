#!/usr/bin/env python
"""
Full-cohort QC for the ico6 metric files produced by convert_metrics_to_ico6.sh.

Per file it records structural checks (vertex count, darray count, map names),
value-integrity checks (NaN/Inf, constant maps), and anatomical plausibility
checks (thickness inside cortex vs medial wall, curv/sulc ranges). Per subject
it also records the left/right vertexwise correlation, which is the check that
caught the mirrored-sphere bug during development.

Cohort-level outliers are flagged by robust z-score (median / MAD) so that a
handful of bad subjects cannot inflate the threshold and hide themselves.

Usage:
    python qc_ico6_metrics.py [-m METRICS_DIR] [-o OUT_CSV]
"""

import argparse
import os
import sys

import numpy as np
import nibabel as nb
import pandas as pd

CHANNELS = ["curv", "sulc", "thickness"]
N_VERTICES = 40962

# Plausibility bounds. Deliberately wide: these flag broken conversions, not
# biological variation. Thickness is FreeSurfer-capped at 5 mm; barycentric
# interpolation lands on 5.0000005 in float32, so the bound needs a tolerance.
BOUNDS = {
    "curv": (-2.5, 2.5),
    "sulc": (-15.0, 15.0),
    "thickness": (0.0, 5.0),
}
BOUND_TOL = 1e-4
CORTEX_THICKNESS_RANGE = (1.8, 3.6)  # mm, cohort-plausible mean

# FreeSurfer ?h.curv carries curvature singularities at a handful of vertices
# (native values reach -300 and beyond). Resampling attenuates but does not
# remove them, and the SVT dataloader is expected to clip them at training time.
# So out-of-range curv is reported as a warning with a vertex count, not an
# error, and only becomes an error if it stops being a sparse spike.
CURV_SPIKE_MAX_VERTICES = 200


def robust_z(values):
    """Median/MAD z-score. Falls back to zeros when MAD collapses."""
    v = np.asarray(values, dtype=float)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad == 0:
        return np.zeros_like(v)
    return 0.6745 * (v - med) / mad


def load_cortex_mask(metrics_dir):
    """Cortex mask on the ico6 mesh, resampled from the fs_LR atlasroi."""
    path = os.path.join(os.path.dirname(metrics_dir), "templates", "atlasroi_ico6.shape.gii")
    if not os.path.isfile(path):
        return None
    return np.asarray(nb.load(path).agg_data()).ravel() > 0.5


def check_file(path, mask):
    """Structural + value checks for one hemisphere file."""
    row = {"file": os.path.basename(path), "errors": [], "warnings": []}
    try:
        gii = nb.load(path)
    except Exception as exc:  # unreadable file is a hard failure
        row["errors"].append(f"unreadable: {exc}")
        return row

    data = np.asarray(gii.agg_data())
    if data.ndim == 1:
        data = data[None, :]

    row["n_darrays"] = data.shape[0]
    row["n_vertices"] = data.shape[1]
    row["map_names"] = ",".join(str(d.meta.get("Name")) for d in gii.darrays)

    if data.shape[0] != len(CHANNELS):
        row["errors"].append(f"expected {len(CHANNELS)} darrays, got {data.shape[0]}")
        return row
    if data.shape[1] != N_VERTICES:
        row["errors"].append(f"expected {N_VERTICES} vertices, got {data.shape[1]}")
        return row
    if row["map_names"] != ",".join(CHANNELS):
        row["warnings"].append(f"unexpected map names: {row['map_names']}")

    for i, name in enumerate(CHANNELS):
        v = data[i]
        n_nan = int(np.isnan(v).sum())
        n_inf = int(np.isinf(v).sum())
        row[f"{name}_nan"] = n_nan
        row[f"{name}_inf"] = n_inf
        if n_nan or n_inf:
            row["errors"].append(f"{name}: {n_nan} NaN, {n_inf} Inf")
            continue

        row[f"{name}_min"] = float(v.min())
        row[f"{name}_max"] = float(v.max())
        row[f"{name}_mean"] = float(v.mean())
        row[f"{name}_std"] = float(v.std())

        if v.std() == 0:
            row["errors"].append(f"{name}: constant map")

        lo, hi = BOUNDS[name]
        n_out = int(((v < lo - BOUND_TOL) | (v > hi + BOUND_TOL)).sum())
        row[f"{name}_n_out_of_bounds"] = n_out
        if n_out:
            msg = f"{name}: {n_out} vertices outside [{lo},{hi}] -> [{v.min():.3f},{v.max():.3f}]"
            if name == "curv" and n_out <= CURV_SPIKE_MAX_VERTICES:
                # sparse FreeSurfer curvature singularity: expected, clipped at train time
                row["warnings"].append(msg + " (sparse curv spike)")
            else:
                row["errors"].append(msg)

        if mask is not None:
            row[f"{name}_cortex_mean"] = float(v[mask].mean())
            if name == "thickness":
                cm = float(v[mask].mean())
                row["thickness_zeros_in_cortex"] = int((v[mask] == 0).sum())
                row["thickness_medialwall_mean"] = float(v[~mask].mean())
                if not (CORTEX_THICKNESS_RANGE[0] <= cm <= CORTEX_THICKNESS_RANGE[1]):
                    row["errors"].append(f"cortex thickness mean {cm:.3f} outside {CORTEX_THICKNESS_RANGE}")
                # A large block of zero-thickness cortex vertices is the signature
                # of a hemisphere/sphere misalignment.
                if row["thickness_zeros_in_cortex"] > 100:
                    row["errors"].append(
                        f"{row['thickness_zeros_in_cortex']} zero-thickness vertices inside cortex"
                    )
    return row


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("-m", "--metrics-dir", default=os.path.join(here, "metrics"))
    ap.add_argument("-o", "--out-csv", default=os.path.join(here, "qc_report.csv"))
    args = ap.parse_args()

    mask = load_cortex_mask(args.metrics_dir)
    print(f"cortex mask: {'loaded' if mask is not None else 'NOT FOUND (cortex checks skipped)'}")

    files = sorted(f for f in os.listdir(args.metrics_dir) if f.endswith(".shape.gii"))
    if not files:
        print("no .shape.gii files found", file=sys.stderr)
        return 1
    print(f"checking {len(files)} files ...")

    rows = []
    for i, fname in enumerate(files, 1):
        rows.append(check_file(os.path.join(args.metrics_dir, fname), mask))
        if i % 50 == 0:
            print(f"  {i}/{len(files)}")

    df = pd.DataFrame(rows)
    df["subject"] = df["file"].str.split(".").str[0]
    df["hemi"] = df["file"].str.split(".").str[1]

    # Left/right vertexwise correlation per subject.
    lr_rows = []
    for subj, grp in df.groupby("subject"):
        if set(grp["hemi"]) != {"L", "R"}:
            lr_rows.append({"subject": subj, "lr_missing_hemi": True})
            continue
        try:
            dl = np.asarray(nb.load(os.path.join(args.metrics_dir, f"{subj}.L.ico6_fs_LR.shape.gii")).agg_data())
            dr = np.asarray(nb.load(os.path.join(args.metrics_dir, f"{subj}.R.ico6_fs_LR.shape.gii")).agg_data())
        except Exception:
            lr_rows.append({"subject": subj, "lr_missing_hemi": True})
            continue
        sel = mask if mask is not None else slice(None)
        rec = {"subject": subj, "lr_missing_hemi": False}
        for i, name in enumerate(CHANNELS):
            a, b = dl[i][sel], dr[i][sel]
            rec[f"lr_corr_{name}"] = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else np.nan
        lr_rows.append(rec)
    lr = pd.DataFrame(lr_rows)

    # Cohort outliers on per-file summary statistics.
    for name in CHANNELS:
        col = f"{name}_cortex_mean" if f"{name}_cortex_mean" in df else f"{name}_mean"
        if col in df:
            valid = df[col].notna()
            df.loc[valid, f"{name}_z"] = robust_z(df.loc[valid, col].values)

    df["n_errors"] = df["errors"].apply(len)
    df["n_warnings"] = df["warnings"].apply(len)
    df["errors"] = df["errors"].apply("; ".join)
    df["warnings"] = df["warnings"].apply("; ".join)

    df = df.merge(lr, on="subject", how="left")
    df.to_csv(args.out_csv, index=False)

    # ---- summary ----
    n_err = int((df["n_errors"] > 0).sum())
    n_warn = int((df["n_warnings"] > 0).sum())
    print("\n" + "=" * 60)
    print(f"files checked      : {len(df)}")
    print(f"files with ERRORS  : {n_err}")
    print(f"files with warnings: {n_warn}")

    if n_err:
        print("\n--- files with errors ---")
        for _, r in df[df["n_errors"] > 0].iterrows():
            print(f"  {r['file']}: {r['errors']}")

    print("\n--- channel summary (cortex mean across cohort) ---")
    for name in CHANNELS:
        col = f"{name}_cortex_mean" if f"{name}_cortex_mean" in df else f"{name}_mean"
        if col in df:
            v = df[col].dropna()
            print(f"  {name:10s} median {v.median():7.3f}  IQR [{v.quantile(.25):7.3f},{v.quantile(.75):7.3f}]"
                  f"  min {v.min():7.3f}  max {v.max():7.3f}")

    print("\n--- L/R vertexwise correlation (cohort) ---")
    for name in CHANNELS:
        c = f"lr_corr_{name}"
        if c in lr:
            v = lr[c].dropna()
            print(f"  {name:10s} median {v.median():.4f}  min {v.min():.4f}  max {v.max():.4f}")

    # sulc L/R correlation is the sharpest misalignment detector
    if "lr_corr_sulc" in lr:
        bad = lr[lr["lr_corr_sulc"] < 0.4]
        if len(bad):
            print(f"\n!! {len(bad)} subjects with sulc L/R corr < 0.4 (possible misalignment):")
            for _, r in bad.iterrows():
                print(f"     {r['subject']}: {r['lr_corr_sulc']:.4f}")
        else:
            print("\n  all subjects have sulc L/R corr >= 0.4")

    outlier_cols = [f"{n}_z" for n in CHANNELS if f"{n}_z" in df]
    if outlier_cols:
        extreme = df[(df[outlier_cols].abs() > 5).any(axis=1)]
        print(f"\n--- robust-z |z|>5 outliers: {len(extreme)} ---")
        for _, r in extreme.iterrows():
            zs = ", ".join(f"{c[:-2]}={r[c]:.1f}" for c in outlier_cols if abs(r[c]) > 5)
            print(f"     {r['file']}: {zs}")

    print(f"\nreport written to {args.out_csv}")
    print("=" * 60)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
