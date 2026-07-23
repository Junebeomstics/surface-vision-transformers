"""Single source of truth for the SiT encoder configuration.

Both pretraining (src/pretraining/pretrain.py) and fine-tuning
(src/finetuning/fine_tune.py) import ENCODER_KWARGS from here so the encoder
they build is guaranteed identical. This matters because fine-tuning loads the
pretrained `encoder.*` weights into its backbone (see fine_tune.py); if the two
sides ever disagreed on ico_grid / embed_dim / etc., the shapes would differ and
the pretrained weights would be silently dropped by load_state_dict(strict=False).

num_channels is set by whatever cortical features are stacked in the .pt files
(e.g. 3 for curvature, sulcal depth, thickness). ico_grid picks the patching
grid: 2 -> 320 patches x 153 vertices (see models/encoder.py ICO_GRID).
"""

# SiT tiny, ico_grid=2 (320 patches x 153 vertices)
ENCODER_KWARGS = dict(
    ico_grid=2,
    num_channels=3,
    embed_dim=192,
    depth=12,
    num_heads=3,
    dim_head=64,
    mlp_ratio=4,
)
