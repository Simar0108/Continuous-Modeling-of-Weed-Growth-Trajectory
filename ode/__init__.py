"""Neural ODE package for plant growth trajectory modeling."""

from .data import (
    GaussianStateTransformer,
    PlantTrackStateDataset,
    PlantTrackDataModule,
    generate_linear_growth_tracks,
    validate_transformer_on_synthetic,
)

__all__ = [
    "GaussianStateTransformer",
    "PlantTrackStateDataset",
    "PlantTrackDataModule",
    "generate_linear_growth_tracks",
    "validate_transformer_on_synthetic",
]
