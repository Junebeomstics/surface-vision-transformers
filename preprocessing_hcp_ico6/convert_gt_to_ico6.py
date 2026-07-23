#!/usr/bin/env python
"""
Convert native FreeSurfer retinotopy ground truth to the ico6 mesh used by the
surface-vision-transformers codebase, vertex-aligned with the input features
produced by convert_metrics_to_ico6.sh.

Source: Retinotopy/data/raw/converted_native/<subject>/native_metrics/
        <subject>.<metric>.<lh|rh>.native.func.gii   (polarAngle, eccentricity,
        pRFsize, R2), on the SAME native mesh as the FreeSurfer curv/sulc/
        thickness (verified: identical vertex count and sphere.reg).

Two properties of the GT force a mask-aware resample:

  1. Invalid vertices are NaN. Plain barycentric resampling propagates NaN into
     every ico6 vertex that draws from a NaN source, needlessly eroding the
     valid region at its boundary.
  2. polarAngle is circular. Barycentric interpolation of raw angles is wrong at
     the 0 deg / 360 deg wrap (e.g. mean(1, 359) = 180, should be ~0).

Both are handled by resampling in a NaN-free, linear space:

    cosPA, sinPA = cos/sin of polarAngle           (circular -> linear)
    validity     = 1.0 where GT valid else 0.0
    all value channels are zero-filled where invalid

After resampling every channel with BARYCENTRIC onto the same
ico6-deformed_to-fsaverage.<H>.surf.gii template used for the inputs, each value
channel is divided by the resampled validity. Because barycentric weights sum to
1, value_filled_ico = sum(w_i v_i over valid) and validity_ico = sum(w_i over
valid), so the ratio is the valid-only weighted average -- the correct
mask-aware interpolation, with no bleed from invalid neighbours.

Output: <subject>.<L|R>.ico6_fs_LR.gt.shape.gii with named maps
    [polarAngle, eccentricity, pRFsize, R2, valid]
polarAngle is atan2(sinPA, cosPA) wrapped to [0, 360); `valid` is 1.0 where the
resampled validity exceeds VALID_THRESHOLD, else 0.0, and every value channel is
set to NaN outside the valid mask so downstream code cannot mistake a zero-fill
for a real measurement.

RH is resampled onto the L template exactly like the input features, giving the
same left/right vertex correspondence.

Usage:
    python convert_gt_to_ico6.py -i sub-100610
    python convert_gt_to_ico6.py -l subjects_gt181.txt -j 8
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import nibabel as nb

HERE = os.path.dirname(os.path.abspath(__file__))
# All roots are env-overridable so the pipeline is relocatable.
REPO = os.environ.get("DR_REPO", "/mnt/scratch/junb/deepRetinotopy")
GT_ROOT = os.environ.get("GT_ROOT", os.path.join(REPO, "Retinotopy/data/raw/converted_native"))
FS_ROOT = os.environ.get("SUBJECTS_ROOT", "/mnt/storage/junb/HCP_Benson_pRF")
SANDBOX = os.environ.get("SANDBOX", os.path.join(REPO, ".sif_sandbox"))
TEMPLATES = os.environ.get("TEMPLATES", os.path.join(HERE, "templates"))

# Any ico6 vertex whose interpolation draws <50% of its weight from valid source
# vertices is treated as invalid.
VALID_THRESHOLD = 0.5

METRICS = ["polarAngle", "eccentricity", "pRFsize", "R2"]
HEMIS = [("lh", "L"), ("rh", "R")]


def surf_dir(subject):
    import glob
    hits = glob.glob(os.path.join(FS_ROOT, subject, "dt-neuro-freesurfer*/output/surf"))
    return hits[0] if hits else None


def load_native(subject, hemi):
    """Return dict of native GT arrays for one hemisphere, or None if missing."""
    d = os.path.join(GT_ROOT, subject, "native_metrics")
    out = {}
    for m in METRICS:
        p = os.path.join(d, f"{subject}.{m}.{hemi}.native.func.gii")
        if not os.path.isfile(p):
            return None
        out[m] = np.asarray(nb.load(p).agg_data(), dtype=np.float64)
    return out


def build_native_channels(gt):
    """Native (n_channels, n_vertices) array + channel names, mask-aware.

    valid = R2 finite AND polarAngle finite AND eccentricity finite. R2 itself
    can be finite where the fit is poor, so validity keys on the retinotopic
    quantities being present, not on any R2 cutoff (thresholding is a training
    decision, kept out of the GT).
    """
    pa = gt["polarAngle"]
    valid = np.isfinite(pa) & np.isfinite(gt["eccentricity"]) & np.isfinite(gt["R2"])

    pa_rad = np.deg2rad(np.where(valid, pa, 0.0))
    channels = {
        "cosPA": np.where(valid, np.cos(pa_rad), 0.0),
        "sinPA": np.where(valid, np.sin(pa_rad), 0.0),
        "eccentricity": np.where(valid, gt["eccentricity"], 0.0),
        "pRFsize": np.where(valid, gt["pRFsize"], 0.0),
        "R2": np.where(valid, gt["R2"], 0.0),
        "validity": valid.astype(np.float64),
    }
    names = list(channels.keys())
    arr = np.stack([channels[n] for n in names], axis=0).astype(np.float32)
    return arr, names


def write_gifti(path, data, names):
    """data: (n_channels, n_vertices) -> multi-darray func.gii."""
    gii = nb.gifti.GiftiImage()
    for i in range(data.shape[0]):
        da = nb.gifti.GiftiDataArray(
            np.ascontiguousarray(data[i]),
            intent="NIFTI_INTENT_NONE",
            datatype="NIFTI_TYPE_FLOAT32",
        )
        if names:
            da.meta["Name"] = names[i]
        gii.add_gifti_data_array(da)
    nb.save(gii, path)


def resample(native_gii, sphere_reg, target_sphere, out_gii):
    subprocess.run(
        ["singularity", "exec", "-B", "/mnt", SANDBOX, "wb_command",
         "-metric-resample", native_gii, sphere_reg, target_sphere,
         "BARYCENTRIC", out_gii],
        check=True, capture_output=True,
    )


def finalise(ico_arr, names):
    """(n_channels, n_vertices) resampled -> final GT (5, n_vertices)."""
    ch = {n: ico_arr[i] for i, n in enumerate(names)}
    validity = ch["validity"]
    valid = validity > VALID_THRESHOLD
    denom = np.where(valid, validity, 1.0)   # avoid divide-by-zero off-mask

    cos = ch["cosPA"] / denom
    sin = ch["sinPA"] / denom
    pa = np.rad2deg(np.arctan2(sin, cos)) % 360.0

    ecc = ch["eccentricity"] / denom
    size = ch["pRFsize"] / denom
    r2 = ch["R2"] / denom

    nan = np.float32("nan")
    out = np.stack([
        np.where(valid, pa, nan),
        np.where(valid, ecc, nan),
        np.where(valid, size, nan),
        np.where(valid, r2, nan),
        valid.astype(np.float32),
    ], axis=0).astype(np.float32)
    return out, ["polarAngle", "eccentricity", "pRFsize", "R2", "valid"]


def convert_subject(subject, out_dir, force=False):
    sd = surf_dir(subject)
    if sd is None:
        print(f"SKIP {subject}: no FreeSurfer surf dir", file=sys.stderr)
        return False

    ok = True
    for hemi, upper in HEMIS:
        out = os.path.join(out_dir, f"{subject}.{upper}.ico6_fs_LR.gt.shape.gii")
        if os.path.isfile(out) and not force:
            print(f"EXISTS {subject} {hemi}")
            continue

        gt = load_native(subject, hemi)
        if gt is None:
            print(f"SKIP {subject} {hemi}: missing native GT", file=sys.stderr)
            ok = False
            continue

        sphere = os.path.join(sd, f"{hemi}.sphere.reg.surf.gii")
        target = os.path.join(TEMPLATES, f"ico6-deformed_to-fsaverage.{upper}.surf.gii")
        for req in (sphere, target):
            if not os.path.isfile(req):
                print(f"SKIP {subject} {hemi}: missing {os.path.basename(req)}", file=sys.stderr)
                ok = False
                break
        else:
            arr, names = build_native_channels(gt)
            with tempfile.TemporaryDirectory() as work:
                nat = os.path.join(work, "native.func.gii")
                ico = os.path.join(work, "ico6.func.gii")
                write_gifti(nat, arr, names)
                resample(nat, sphere, target, ico)
                ico_arr = np.asarray(nb.load(ico).agg_data(), dtype=np.float64)
                final, fnames = finalise(ico_arr, names)
                write_gifti(out, final, fnames)
            nvalid = int(final[4].sum())
            print(f"OK {subject} {hemi} -> {os.path.basename(out)} ({nvalid} valid)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--subject")
    ap.add_argument("-l", "--list")
    ap.add_argument("-o", "--out-dir", default=os.path.join(HERE, "gt"))
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("-f", "--force", action="store_true")
    args = ap.parse_args()

    if not args.subject and not args.list:
        ap.error("provide -i <subject> or -l <list>")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.subject:
        subjects = [args.subject]
    else:
        with open(args.list) as fh:
            subjects = [ln.strip() for ln in fh if ln.strip()]

    if args.jobs > 1 and len(subjects) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        fails = 0
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(convert_subject, s, args.out_dir, args.force): s
                    for s in subjects}
            for fut in as_completed(futs):
                if not fut.result():
                    fails += 1
        print(f"\ndone. {len(subjects)-fails}/{len(subjects)} subjects ok")
        return 1 if fails else 0
    else:
        fails = sum(0 if convert_subject(s, args.out_dir, args.force) else 1
                    for s in subjects)
        return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
