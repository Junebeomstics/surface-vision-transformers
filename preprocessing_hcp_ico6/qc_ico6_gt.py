#!/usr/bin/env python
"""
Full-cohort QC for the ico6 ground-truth files from convert_gt_to_ico6.py.

Checks per file: structure (5 maps, 40962 vertices, map names), value integrity
(off-mask must be NaN, on-mask must be finite), and plausibility (polarAngle in
[0,360), eccentricity/pRFsize/R2 non-negative, R2 on a ~0-100 scale). Per
subject it reports the left/right valid-fraction agreement and the polarAngle
circular correlation between hemispheres as a coarse alignment check.

Usage: python qc_ico6_gt.py [-g GT_DIR] [-o OUT_CSV]
"""

import argparse
import os
import sys

import numpy as np
import nibabel as nb
import pandas as pd

N_VERTICES = 40962
MAPS = ["polarAngle", "eccentricity", "pRFsize", "R2", "valid"]


def circ_corr(a, b):
    """Circular correlation of two angle arrays in degrees (Fisher-Lee)."""
    a = np.deg2rad(a)
    b = np.deg2rad(b)
    a_ = a - np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    b_ = b - np.arctan2(np.sin(b).mean(), np.cos(b).mean())
    num = np.sum(np.sin(a_) * np.sin(b_))
    den = np.sqrt(np.sum(np.sin(a_) ** 2) * np.sum(np.sin(b_) ** 2))
    return float(num / den) if den > 0 else np.nan


def check_file(path):
    row = {"file": os.path.basename(path), "errors": [], "warnings": []}
    try:
        gii = nb.load(path)
    except Exception as exc:
        row["errors"].append(f"unreadable: {exc}")
        return row

    data = np.asarray(gii.agg_data(), dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    row["n_maps"] = data.shape[0]
    row["n_vertices"] = data.shape[1]
    row["map_names"] = ",".join(str(d.meta.get("Name")) for d in gii.darrays)

    if data.shape[0] != len(MAPS):
        row["errors"].append(f"expected {len(MAPS)} maps, got {data.shape[0]}")
        return row
    if data.shape[1] != N_VERTICES:
        row["errors"].append(f"expected {N_VERTICES} vertices, got {data.shape[1]}")
        return row
    if row["map_names"] != ",".join(MAPS):
        row["warnings"].append(f"unexpected map names: {row['map_names']}")

    valid = data[4] > 0.5
    row["n_valid"] = int(valid.sum())
    row["valid_frac"] = float(valid.mean())

    if row["n_valid"] == 0:
        row["errors"].append("no valid vertices")
        return row

    # valid map must be exactly 0/1
    uniq = np.unique(data[4])
    if not np.all(np.isin(uniq, [0.0, 1.0])):
        row["errors"].append(f"valid map not binary: {uniq[:5]}")

    for i, name in enumerate(MAPS[:4]):
        on = data[i][valid]
        off = data[i][~valid]
        # on-mask must be finite
        n_bad = int((~np.isfinite(on)).sum())
        if n_bad:
            row["errors"].append(f"{name}: {n_bad} non-finite inside valid mask")
            continue
        # off-mask must be NaN (so a zero-fill can't be read as a measurement)
        if not np.all(np.isnan(off)):
            n_notnan = int((~np.isnan(off)).sum())
            row["errors"].append(f"{name}: {n_notnan} off-mask vertices are not NaN")

        row[f"{name}_min"] = float(on.min())
        row[f"{name}_max"] = float(on.max())
        row[f"{name}_mean"] = float(on.mean())

    # plausibility
    if "polarAngle_min" in row:
        if row["polarAngle_min"] < 0 or row["polarAngle_max"] >= 360.0 + 1e-3:
            row["errors"].append(
                f"polarAngle outside [0,360): [{row['polarAngle_min']:.2f},{row['polarAngle_max']:.2f}]")
    for name in ("eccentricity", "pRFsize", "R2"):
        if f"{name}_min" in row and row[f"{name}_min"] < -1e-3:
            row["errors"].append(f"{name}: negative min {row[f'{name}_min']:.3f}")
    if "R2_max" in row and row["R2_max"] <= 1.5:
        # convert_gt keeps native R2 scale (~0-100); a <=1 max means the scale
        # assumption is wrong and any R2>10 threshold downstream would be empty
        row["warnings"].append(f"R2 max {row['R2_max']:.3f} looks 0-1 scaled, expected 0-100")
    return row


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("-g", "--gt-dir", default=os.path.join(here, "gt"))
    ap.add_argument("-o", "--out-csv", default=os.path.join(here, "qc_gt_report.csv"))
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.gt_dir) if f.endswith(".gt.shape.gii"))
    if not files:
        print("no GT files found", file=sys.stderr)
        return 1
    print(f"checking {len(files)} GT files ...")

    rows = []
    for i, f in enumerate(files, 1):
        rows.append(check_file(os.path.join(args.gt_dir, f)))
        if i % 50 == 0:
            print(f"  {i}/{len(files)}")

    df = pd.DataFrame(rows)
    df["subject"] = df["file"].str.split(".").str[0]
    df["hemi"] = df["file"].str.split(".").str[1]

    # per-subject L/R checks
    lr = []
    for subj, grp in df.groupby("subject"):
        if set(grp["hemi"]) != {"L", "R"}:
            lr.append({"subject": subj, "lr_ok": False})
            continue
        rec = {"subject": subj, "lr_ok": True}
        try:
            dl = np.asarray(nb.load(os.path.join(args.gt_dir, f"{subj}.L.ico6_fs_LR.gt.shape.gii")).agg_data())
            dr = np.asarray(nb.load(os.path.join(args.gt_dir, f"{subj}.R.ico6_fs_LR.gt.shape.gii")).agg_data())
            both = (dl[4] > 0.5) & (dr[4] > 0.5)
            rec["lr_shared_valid"] = int(both.sum())
            rec["lr_polarAngle_circcorr"] = circ_corr(dl[0][both], dr[0][both])
        except Exception as exc:
            rec["lr_ok"] = False
            rec["err"] = str(exc)
        lr.append(rec)
    lrdf = pd.DataFrame(lr)

    df["n_errors"] = df["errors"].apply(len)
    df["n_warnings"] = df["warnings"].apply(len)
    df["errors"] = df["errors"].apply("; ".join)
    df["warnings"] = df["warnings"].apply("; ".join)
    df = df.merge(lrdf, on="subject", how="left")
    df.to_csv(args.out_csv, index=False)

    n_err = int((df["n_errors"] > 0).sum())
    print("\n" + "=" * 60)
    print(f"files checked     : {len(df)}")
    print(f"files with ERRORS : {n_err}")
    print(f"files with warns  : {int((df['n_warnings'] > 0).sum())}")
    if n_err:
        print("\n--- errors ---")
        for _, r in df[df["n_errors"] > 0].iterrows():
            print(f"  {r['file']}: {r['errors']}")

    print("\n--- valid fraction (cohort) ---")
    v = df["valid_frac"].dropna()
    print(f"  median {v.median():.3f}  min {v.min():.3f}  max {v.max():.3f}")

    print("\n--- value ranges (cohort, on-mask means) ---")
    for name in MAPS[:4]:
        c = f"{name}_mean"
        if c in df:
            vv = df[c].dropna()
            print(f"  {name:12s} mean-of-means {vv.mean():8.3f}  [{vv.min():8.3f}, {vv.max():8.3f}]")

    if "lr_polarAngle_circcorr" in lrdf:
        cc = lrdf["lr_polarAngle_circcorr"].dropna()
        print(f"\n--- L/R polarAngle circular corr: median {cc.median():.3f} "
              f"min {cc.min():.3f} max {cc.max():.3f} ---")

    print(f"\nreport: {args.out_csv}")
    print("=" * 60)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
