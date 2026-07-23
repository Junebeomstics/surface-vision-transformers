#!/usr/bin/env python
"""
Build ico6 visual-area ROI masks from the Wang et al. (2015) atlas used by
deepRetinotopy, and derive the roi_vertex_index_{hemisphere}.pt files that
SiTRetinotopy (src/finetuning/sit_dataset.py) expects.

deepRetinotopy defines its ROI natively on the HCP 32k fs_LR group mesh
(32492 vertices/hemisphere), not on the ico6 mesh this repo trains on. So
the mask is converted .mat -> .shape.gii on 32k_fs_LR, then resampled onto
ico6 with wb_command (same BARYCENTRIC pattern already used to build
preprocessing_hcp_ico6/templates/atlasroi_ico6.shape.gii), then thresholded
and reduced to a vertex-index LongTensor.

Two variants are produced (both hemispheres each):
  - v1v3fovea:        V1d,V1v,V2d,V2v,V3d,V3v + fovea_V1/V2/V3
                       (== ROI_WangPlusFovea/ROI.mat, deepRetinotopy's
                       standard training ROI)
  - fullvisualcortex:  v1v3fovea + hV4,VO1,VO2,PHC1,PHC2,V3a,V3b,LO1,LO2,
                       TO1,TO2,IPS0-5,SPL1 (all Wang2015 areas)

Requires: scipy, nibabel (use e.g. `conda run -n eigenmode_exp python ...`),
and singularity + the deepRetinotopy wb_command sandbox.

Usage:
    python build_roi_visual_areas.py
"""
import os
import os.path as osp
import subprocess

import numpy as np
import nibabel as nb
import scipy.io
import torch

HERE = osp.dirname(osp.abspath(__file__))
SVT_ROOT = osp.dirname(HERE)

DR_LABELS = os.environ.get(
    "DR_LABELS", "/mnt/scratch/junb/deepRetinotopy_Ribeiro/Retinotopy/labels")
ATLASES = os.environ.get(
    "ATLASES", "/mnt/scratch/junb/HCPpipelines/global/templates/standard_mesh_atlases")
# Per-hemisphere ico6 target spheres the DATA (metrics/gt) was resampled onto
# (convert_gt_to_ico6.py: target = ico6-deformed_to-fsaverage.<H>.surf.gii). The
# ROI MUST resample onto the SAME per-hemisphere target, paired with a 32k source
# sphere in the same deformed-to-fsaverage frame -- otherwise the R hemisphere
# mask lands on L geometry and picks the wrong vertices (L/R ico6 spheres differ).
TEMPLATES = osp.join(HERE, "templates")
RESAMPLE_FSAVERAGE = osp.join(ATLASES, "resample_fsaverage")
SANDBOX = os.environ.get("SANDBOX", "/mnt/scratch/junb/deepRetinotopy/.sif_sandbox")

OUT_DIR = osp.join(SVT_ROOT, "data", "finetuning")
NUM_HEMI_NODES = 32492  # 32k_fs_LR vertices per hemisphere

# V1-V3 + fovea patches: identical set to ROI_WangPlusFovea/ROI.mat
V1V3_FOVEA_LABELS = [
    "V1d", "V1v", "fovea_V1",
    "V2d", "V2v", "fovea_V2",
    "V3d", "V3v", "fovea_V3",
]
# Every parcel in the Wang et al. (2015) atlas
FULL_VISUAL_CORTEX_LABELS = V1V3_FOVEA_LABELS + [
    "hV4", "VO1", "VO2", "PHC1", "PHC2", "V3a", "V3b",
    "LO1", "LO2", "TO1", "TO2",
    "IPS0", "IPS1", "IPS2", "IPS3", "IPS4", "IPS5", "SPL1",
]

VARIANTS = {
    "v1v3fovea": V1V3_FOVEA_LABELS,
    "fullvisualcortex": FULL_VISUAL_CORTEX_LABELS,
}

HEMIS = [("L", 0, NUM_HEMI_NODES), ("R", NUM_HEMI_NODES, 2 * NUM_HEMI_NODES)]


def load_wang_mask_64984(labels):
    """Sum the named Wang2015 label .mat files into a (64984,) binary mask."""
    total = np.zeros(64984)
    for label in labels:
        mat = scipy.io.loadmat(
            osp.join(DR_LABELS, "VisualAreasLabels_Wang2015", f"{label}_labels.mat"))
        total += np.reshape(mat[label][0:64984], (-1))
    return (total >= 1).astype(np.float32)


def write_shape_gii(path, data):
    gii = nb.gifti.GiftiImage()
    da = nb.gifti.GiftiDataArray(
        np.ascontiguousarray(data.astype(np.float32)),
        intent="NIFTI_INTENT_NONE", datatype="NIFTI_TYPE_FLOAT32")
    gii.add_gifti_data_array(da)
    nb.save(gii, path)


def wb_resample(metric_in, current_sphere, new_sphere, metric_out):
    subprocess.run(
        ["singularity", "exec", "-B", "/mnt", SANDBOX, "wb_command",
         "-metric-resample", metric_in, current_sphere, new_sphere,
         "BARYCENTRIC", metric_out],
        check=True, capture_output=True,
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    work_dir = osp.join(OUT_DIR, "_roi_build_work")
    os.makedirs(work_dir, exist_ok=True)

    for variant_name, labels in VARIANTS.items():
        mask_64984 = load_wang_mask_64984(labels)
        print(f"[{variant_name}] fs_LR 32k mask: "
              f"L={int(mask_64984[:NUM_HEMI_NODES].sum())} "
              f"R={int(mask_64984[NUM_HEMI_NODES:].sum())} nonzero vertices")

        for hemi, lo, hi in HEMIS:
            mask_hemi = mask_64984[lo:hi]  # (32492,)

            fslr_gii = osp.join(work_dir, f"wang_{variant_name}.{hemi}.32k_fs_LR.shape.gii")
            write_shape_gii(fslr_gii, mask_hemi)

            current_sphere = osp.join(
                RESAMPLE_FSAVERAGE,
                f"fs_LR-deformed_to-fsaverage.{hemi}.sphere.32k_fs_LR.surf.gii")
            target_sphere = osp.join(TEMPLATES, f"ico6-deformed_to-fsaverage.{hemi}.surf.gii")
            ico6_gii = osp.join(work_dir, f"wang_{variant_name}.{hemi}.ico6.shape.gii")
            wb_resample(fslr_gii, current_sphere, target_sphere, ico6_gii)

            resampled = np.asarray(nb.load(ico6_gii).agg_data(), dtype=np.float64)
            binary = resampled > 0.5
            index = torch.as_tensor(np.nonzero(binary)[0], dtype=torch.long)

            hemi_name = {"L": "Left", "R": "Right"}[hemi]
            out_pt = osp.join(OUT_DIR, f"roi_vertex_index_{hemi_name}_{variant_name}.pt")
            torch.save(index, out_pt)

            out_gii = osp.join(OUT_DIR, f"roi_mask_{hemi_name}_{variant_name}.ico6.shape.gii")
            write_shape_gii(out_gii, binary.astype(np.float32))

            print(f"[{variant_name}] {hemi_name}: {index.numel()} ico6 vertices "
                  f"-> {out_pt}")


if __name__ == "__main__":
    main()
