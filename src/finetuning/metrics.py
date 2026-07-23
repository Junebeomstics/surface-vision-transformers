"""Loss and evaluation metrics for retinotopy regression.

Two things the plain R2-weighted Smooth-L1 in fine_tune.py does not handle:

1. polarAngle is circular (0 deg == 360 deg). Smooth-L1 on the raw angle is
   discontinuous at the wrap -- a prediction of 359 deg for a target of 1 deg is
   penalised as if it were 358 deg off, not 2 deg. `masked_circular_loss` uses
   1 - cos(pred - target) instead, which is periodic and correct at the wrap.
   eccentricity and pRFsize are not circular, so they keep Smooth-L1.

2. MAE alone under-describes retinotopy quality, and for a circular quantity it
   is not even meaningful. `regression_metrics` reports, per project policy:
     polarAngle   -> circular correlation (astropy) as primary,
                     plus Pearson/Spearman on wrapped degrees for comparability
     eccentricity -> Pearson + Spearman
     pRFsize      -> Pearson + Spearman (Spearman is the primary for pRFsize)

Both operate in full-mesh vertex space with an R2 vector that is zero outside
the ROI (see sit_dataset.py); supervision/evaluation is therefore ROI-restricted
without an explicit mask, exactly as the R2-weighted Smooth-L1 already was.
Angles are in degrees throughout (the RetinoSolver ROI .pt files store
polarAngle in [0, 360)).
"""
import numpy as np
import torch

from astropy.stats import circcorrcoef
from astropy import units as u
from scipy.stats import pearsonr, spearmanr

CIRCULAR = {"polarAngle"}
# pRFsize is judged primarily by Spearman; the others by their natural primary.
PRIMARY_METRIC = {
    "polarAngle": "circ_corr",
    "eccentricity": "pearson",
    "pRFsize": "spearman",
}


def masked_circular_loss(pred_deg, target_deg, r2):
    """R2-weighted circular loss, angles in degrees.

    sum(r2 * (1 - cos(pred - target))) / sum(r2). r2 is zero outside the ROI,
    so this restricts supervision to the ROI just like the Smooth-L1 path.
    """
    diff = torch.deg2rad(pred_deg - target_deg)
    per_vertex = 1.0 - torch.cos(diff)
    denom = r2.sum().clamp(min=1e-6)
    return (r2 * per_vertex).sum() / denom


def _circular_correlation(pred_deg, target_deg):
    """Fisher-Lee circular correlation via astropy, inputs in degrees."""
    pred = np.deg2rad(np.mod(pred_deg, 360.0))
    target = np.deg2rad(np.mod(target_deg, 360.0))
    return float(circcorrcoef(pred * u.rad, target * u.rad))


def regression_metrics(pred, target, r2, prediction, r2_thr):
    """Metrics over vertices with R2 > r2_thr. pred/target/r2 are 1-D arrays."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    r2 = np.asarray(r2, dtype=np.float64).ravel()

    mask = r2 > r2_thr
    if mask.sum() < 2:
        return {}
    p, t = pred[mask], target[mask]

    if prediction in CIRCULAR:
        pw, tw = np.mod(p, 360.0), np.mod(t, 360.0)
        return {
            "circ_corr": _circular_correlation(pw, tw),
            "pearson": float(pearsonr(pw, tw)[0]),
            "spearman": float(spearmanr(pw, tw)[0]),
        }
    return {
        "pearson": float(pearsonr(p, t)[0]),
        "spearman": float(spearmanr(p, t)[0]),
    }
