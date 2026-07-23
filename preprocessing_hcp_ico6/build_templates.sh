#!/usr/bin/env bash
#
# Rebuild the ico6 resample-target templates (Phase 0 of the pipeline).
#
# These map the surface-vision-transformers ico6 mesh (utils/ico-6-L.surf.gii)
# into fsaverage coordinates so a single ADAP_BARY_AREA resample takes a native
# metric straight to ico6. They are checked in under templates/, so you only
# need this to reproduce them.
#
# Requires the fs_LR standard-mesh atlases (HCPpipelines global templates) and
# the SVT ico6 mesh.
#
# ENV OVERRIDES:
#   SANDBOX     /mnt/scratch/junb/deepRetinotopy/.sif_sandbox
#   ATLASES     /mnt/scratch/junb/HCPpipelines/global/templates/standard_mesh_atlases
#   ICO6_MESH   <svt repo>/utils/ico-6-L.surf.gii

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SVT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SANDBOX="${SANDBOX:-/mnt/scratch/junb/deepRetinotopy/.sif_sandbox}"
ATLASES="${ATLASES:-/mnt/scratch/junb/HCPpipelines/global/templates/standard_mesh_atlases}"
ICO6_MESH="${ICO6_MESH:-$SVT_ROOT/utils/ico-6-L.surf.gii}"
OUT="$SCRIPT_DIR/templates"
mkdir -p "$OUT"

# project-to is ALWAYS the L fs_LR sphere: ico-6-L lives in the L frame, and
# fs_LR vertex i of L corresponds anatomically to vertex i of R, so only the
# unproject-from sphere switches hemisphere. Using the R sphere as project-to
# breaks L/R vertex correspondence (fs_LR L/R spheres are exact x-mirrors).
for H in L R; do
    singularity exec -B /mnt "$SANDBOX" wb_command -surface-sphere-project-unproject \
        "$ICO6_MESH" \
        "$ATLASES/L.sphere.164k_fs_LR.surf.gii" \
        "$ATLASES/resample_fsaverage/fs_LR-deformed_to-fsaverage.$H.sphere.164k_fs_LR.surf.gii" \
        "$OUT/ico6-deformed_to-fsaverage.$H.surf.gii"
    echo "built $OUT/ico6-deformed_to-fsaverage.$H.surf.gii"
done

# cortex mask on ico6, resampled from the fs_LR L atlasroi
singularity exec -B /mnt "$SANDBOX" wb_command -metric-resample \
    "$ATLASES/L.atlasroi.164k_fs_LR.shape.gii" \
    "$ATLASES/L.sphere.164k_fs_LR.surf.gii" \
    "$ICO6_MESH" BARYCENTRIC "$OUT/atlasroi_ico6.shape.gii"
echo "built $OUT/atlasroi_ico6.shape.gii"
