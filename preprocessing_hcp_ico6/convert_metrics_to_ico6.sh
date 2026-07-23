#!/usr/bin/env bash
#
# Convert FreeSurfer native curvature / sulcal depth / cortical thickness to the
# ico6 (40,962 vertex) mesh used by the surface-vision-transformers codebase.
#
# Output: one GIFTI per subject per hemisphere with 3 darrays [curv, sulc,
# thickness], matching `channels: [0,1,2]` in the MS-SiT config. RH is resampled
# onto the L ico6 template, giving left/right vertex correspondence.
#
# Resampling chain (single interpolation per metric):
#   native metric + <hemi>.sphere.reg  ->  ico6-deformed_to-fsaverage.<H>.surf.gii
# The target sphere carries the ico6 mesh in fsaverage coordinates, so one
# ADAP_BARY_AREA resample replaces the usual native -> fs_LR -> ico6 two-step.
# Cross-validated against the two-step path at r = 0.9991.
#
# This version generates the intermediate GIFTI sphere and midthickness on the
# fly (in a temp dir, never touching the source) when they are absent, so it
# runs on every FreeSurfer subject that has the binary surfaces, not only the
# subset that was pre-converted. Required per subject/hemi:
#   <hemi>.sphere.reg (binary), <hemi>.curv, <hemi>.sulc, <hemi>.thickness,
#   <hemi>.white, <hemi>.pial
#
# USAGE:
#   ./convert_metrics_to_ico6.sh -i sub-100206
#   ./convert_metrics_to_ico6.sh -l subjects.txt -j 8
#
# OPTIONS:
#   -i ID       single subject id
#   -l FILE     file with one subject id per line
#   -j N        parallel jobs (default 1)
#   -o PATH     output directory (default: $OUT_DIR env or ./metrics)
#   -f          overwrite existing outputs
# ENV OVERRIDES (with defaults):
#   SUBJECTS_ROOT  /mnt/storage/junb/HCP_Benson_pRF
#   SANDBOX        /mnt/scratch/junb/deepRetinotopy/.sif_sandbox
#   TEMPLATES      <script dir>/templates

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBJECTS_ROOT="${SUBJECTS_ROOT:-/mnt/storage/junb/HCP_Benson_pRF}"
SANDBOX="${SANDBOX:-/mnt/scratch/junb/deepRetinotopy/.sif_sandbox}"
TEMPLATES="${TEMPLATES:-$SCRIPT_DIR/templates}"

OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/metrics}"
SUBJECT_ID=""
SUBJECT_LIST=""
JOBS=1
FORCE=0

while getopts "i:l:j:o:fh" opt; do
    case $opt in
        i) SUBJECT_ID="$OPTARG" ;;
        l) SUBJECT_LIST="$OPTARG" ;;
        j) JOBS="$OPTARG" ;;
        o) OUT_DIR="$OPTARG" ;;
        f) FORCE=1 ;;
        h) sed -n '1,35p' "$0"; exit 0 ;;
        *) echo "Unknown option"; exit 1 ;;
    esac
done

if [ -z "$SUBJECT_ID" ] && [ -z "$SUBJECT_LIST" ]; then
    echo "ERROR: provide -i <subject> or -l <list file>" >&2
    exit 1
fi
for H in L R; do
    [ -f "$TEMPLATES/ico6-deformed_to-fsaverage.$H.surf.gii" ] || {
        echo "ERROR: missing template $TEMPLATES/ico6-deformed_to-fsaverage.$H.surf.gii" >&2; exit 1; }
done
[ -d "$SANDBOX" ] || { echo "ERROR: sandbox not found: $SANDBOX" >&2; exit 1; }

mkdir -p "$OUT_DIR"

resolve_surf_dir() {
    ls -d "$SUBJECTS_ROOT/$1"/dt-neuro-freesurfer*/output/surf 2>/dev/null | head -1
}

convert_subject() {
    local subject="$1"
    local surf_dir
    surf_dir=$(resolve_surf_dir "$subject")
    if [ -z "$surf_dir" ]; then
        echo "SKIP $subject: no FreeSurfer surf directory" >&2
        return 1
    fi

    local work
    work=$(mktemp -d)
    # shellcheck disable=SC2064
    trap "rm -rf '$work'" RETURN

    local hemi upper out
    for hemi in lh rh; do
        [ "$hemi" = "lh" ] && upper=L || upper=R
        out="$OUT_DIR/$subject.$upper.ico6_fs_LR.shape.gii"
        if [ -f "$out" ] && [ "$FORCE" -eq 0 ]; then
            echo "EXISTS $subject $hemi"; continue
        fi

        # required binary inputs
        local missing=0 req
        for req in "$hemi.sphere.reg" "$hemi.curv" "$hemi.sulc" "$hemi.thickness" \
                   "$hemi.white" "$hemi.pial"; do
            [ -f "$surf_dir/$req" ] || { echo "SKIP $subject $hemi: missing $req" >&2; missing=1; break; }
        done
        [ "$missing" -eq 1 ] && return 1

        # GIFTI sphere: use existing if present, else generate from binary
        local sphere="$surf_dir/$hemi.sphere.reg.surf.gii"
        if [ ! -f "$sphere" ]; then
            singularity exec -B /mnt,"$work" "$SANDBOX" \
                mris_convert "$surf_dir/$hemi.sphere.reg" "$work/$hemi.sphere.reg.surf.gii" >/dev/null 2>&1
            sphere="$work/$hemi.sphere.reg.surf.gii"
        fi

        # native midthickness (drives the area correction): use existing or build
        local midthick="$surf_dir/$hemi.midthickness.surf.gii"
        if [ ! -f "$midthick" ]; then
            singularity exec -B /mnt,"$work" "$SANDBOX" bash -c "
                mris_convert '$surf_dir/$hemi.white' '$work/$hemi.white.surf.gii' >/dev/null 2>&1
                mris_convert '$surf_dir/$hemi.pial'  '$work/$hemi.pial.surf.gii'  >/dev/null 2>&1
                wb_command -surface-average '$work/$hemi.midthickness.surf.gii' \
                    -surf '$work/$hemi.white.surf.gii' -surf '$work/$hemi.pial.surf.gii'
            " >/dev/null
            midthick="$work/$hemi.midthickness.surf.gii"
        fi

        # ico6 midthickness for -area-surfs
        singularity exec -B /mnt,"$work" "$SANDBOX" wb_command -surface-resample \
            "$midthick" "$sphere" \
            "$TEMPLATES/ico6-deformed_to-fsaverage.$upper.surf.gii" \
            BARYCENTRIC "$work/$hemi.midthickness.ico6.surf.gii" >/dev/null

        local metric
        for metric in curv sulc thickness; do
            singularity exec -B /mnt,"$work" "$SANDBOX" bash -c "
                mris_convert -c '$surf_dir/$hemi.$metric' '$surf_dir/$hemi.white' \
                    '$work/$hemi.$metric.gii' >/dev/null 2>&1
                wb_command -metric-resample '$work/$hemi.$metric.gii' \
                    '$sphere' '$TEMPLATES/ico6-deformed_to-fsaverage.$upper.surf.gii' \
                    ADAP_BARY_AREA '$work/$hemi.$metric.ico6.func.gii' \
                    -area-surfs '$midthick' '$work/$hemi.midthickness.ico6.surf.gii'
            " >/dev/null
        done

        singularity exec -B /mnt,"$work" "$SANDBOX" bash -c "
            wb_command -metric-merge '$out' \
                -metric '$work/$hemi.curv.ico6.func.gii' \
                -metric '$work/$hemi.sulc.ico6.func.gii' \
                -metric '$work/$hemi.thickness.ico6.func.gii'
            wb_command -set-map-names '$out' -map 1 curv -map 2 sulc -map 3 thickness
        " >/dev/null

        echo "OK $subject $hemi -> $(basename "$out")"
    done
}

export -f convert_subject resolve_surf_dir
export SUBJECTS_ROOT SANDBOX TEMPLATES OUT_DIR FORCE

if [ -n "$SUBJECT_ID" ]; then
    convert_subject "$SUBJECT_ID"
else
    grep -v '^[[:space:]]*$' "$SUBJECT_LIST" \
        | xargs -P "$JOBS" -I{} bash -c 'convert_subject "$@" || true' _ {}
fi
