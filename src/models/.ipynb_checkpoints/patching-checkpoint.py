"""Mesh <-> patch conversions for the icosphere patching grid.

`triangle_indices_ico_{mesh}_sub_ico_{grid}.csv` (bundled under
data/patch_extraction/) is a static, subject-independent lookup table from
the reference repo's patch_extraction pipeline: column j lists the full-mesh
vertex indices belonging to patch j. Patches overlap at their boundaries, so
going from patches back to mesh vertices needs averaging, not a plain
scatter.

`load_patch_indices`'s `reorder` option applies `order_ico{grid}.npy`, a
locality-preserving reordering of patch indices. That only matters for
windowed/hierarchical models (e.g. MS-SiT, where consecutive patch indices
must be spatially local for windowed attention and patch merging to be
geometrically meaningful) -- SiT's global attention has no such requirement,
so `reorder=False` is the right default there.
"""
import os.path as osp

import numpy as np
import pandas as pd
import torch

DEFAULT_ROOT = osp.join(osp.dirname(__file__), "..", "..", "data", "patch_extraction")


def load_patch_indices(ico_mesh=6, ico_grid=4, reorder=True, root=DEFAULT_ROOT):
    """Returns a LongTensor (num_patches, num_vertices) of full-mesh vertex
    indices, one row per patch, in patch order."""
    path = osp.join(root, f"triangle_indices_ico_{ico_mesh}_sub_ico_{ico_grid}.csv")
    indices = pd.read_csv(path)

    if reorder:
        order = np.load(osp.join(root, "reorder_patches", f"order_ico{ico_grid}.npy"))
        indices = indices[[str(i) for i in order]]

    # columns are patches, rows are the vertices within each patch
    return torch.as_tensor(indices.to_numpy().T, dtype=torch.long)


def patchify(mesh_features, patch_indices):
    """Gather per-vertex mesh features onto the patch grid.

    mesh_features: (..., num_mesh_vertices)
    patch_indices: (num_patches, num_vertices)
    returns: (..., num_patches, num_vertices)
    """
    return mesh_features[..., patch_indices]


def unpatchify_mean(patch_values, patch_indices, num_mesh_vertices):
    """Inverse of `patchify`: scatter patch-level values back onto mesh
    vertices, averaging over patches that share a vertex.

    patch_values: (..., num_patches, num_vertices)
    patch_indices: (num_patches, num_vertices)
    returns: (..., num_mesh_vertices)
    """
    flat_indices = patch_indices.reshape(-1).to(patch_values.device)
    flat_values = patch_values.reshape(*patch_values.shape[:-2], -1)

    out_shape = (*patch_values.shape[:-2], num_mesh_vertices)
    summed = torch.zeros(out_shape, dtype=patch_values.dtype, device=patch_values.device)
    summed.index_add_(-1, flat_indices, flat_values)

    counts = torch.zeros(num_mesh_vertices, dtype=patch_values.dtype, device=patch_values.device)
    counts.index_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=patch_values.dtype))

    return summed / counts.clamp(min=1)
