#!/usr/bin/env python3
"""CPU-only forward-pass check for the H1 discrete baselines.

Does not train and does not import Lightning.  Instantiates LSTM, GRU, and
Transformer against the same PlantTrackStateDataset used by H1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ode.baselines import build_baseline_model
from ode.data.datamodule import _collate_variable_length_tracks
from ode.data.dataset import PlantTrackStateDataset
from ode.data.transforms import GaussianStateTransformer

PARQUET = REPO / "metrics_with_features.parquet"
TARGET_DIM = 3


def _one_batch(n_tracks: int = 4) -> dict:
    df = pd.read_parquet(PARQUET)
    if "valid_track" in df.columns:
        df = df[df["valid_track"]].copy()
    if "species" in df.columns:
        df = df[df["species"] == "Maize"].copy()
    counts = df.groupby("track_id").size()
    track_ids = counts.nlargest(n_tracks).index.tolist()
    train_df = df[df["track_id"].isin(track_ids)]
    transformer = GaussianStateTransformer(sigma_mode="z_score")
    transformer.fit(train_df)
    ds = PlantTrackStateDataset(
        PARQUET,
        transformer,
        track_ids=track_ids,
        dataframe=df,
        valid_track_only=True,
        min_observations=15,
    )
    items = [ds[i] for i in range(len(ds))]
    print(f"[smoke_baselines] loaded {len(items)} maize tracks from {PARQUET.name}")
    return _collate_variable_length_tracks(items) if len(items) > 1 else items[0]


def main() -> None:
    if not PARQUET.is_file():
        raise FileNotFoundError(PARQUET)
    batch = _one_batch()
    states = batch["states_5d"]
    t_abs = batch["t_absolute"]
    lengths = batch["track_lengths"]
    k = 3
    horizon = states.shape[1] - k
    print(
        f"[smoke_baselines] batch states={tuple(states.shape)}  "
        f"horizon={horizon}  lengths={lengths.tolist()}"
    )

    for name in ("lstm", "gru", "transformer"):
        model = build_baseline_model(name, n_context_frames=k)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        with torch.inference_mode():
            tf = model.teacher_forcing_rollout(states, t_abs, lengths)
            ar = model.autoregressive_rollout(states, t_abs, lengths)
        assert tf.shape == (states.shape[0], horizon, TARGET_DIM), (
            f"{name} teacher-force shape {tuple(tf.shape)} "
            f"!= {(states.shape[0], horizon, TARGET_DIM)}"
        )
        assert ar.shape == tf.shape, f"{name} AR shape {tuple(ar.shape)} != TF {tuple(tf.shape)}"
        assert torch.isfinite(tf).all(), f"{name} teacher-force produced non-finite values"
        assert torch.isfinite(ar).all(), f"{name} AR produced non-finite values"
        print(
            f"  PASS {name:12s}  params={n_params:7d}  "
            f"tf={tuple(tf.shape)}  ar={tuple(ar.shape)}",
            flush=True,
        )
    print("[smoke_baselines] all three baselines forward on the H1 dataloader", flush=True)


if __name__ == "__main__":
    main()
