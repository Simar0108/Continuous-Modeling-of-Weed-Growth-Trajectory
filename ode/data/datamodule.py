"""
PlantTrackDataModule: Lightning-style data module.

Splits by tray_id stratified by coverage hours; fits GaussianStateTransformer
on train set only; provides train/val/test dataloaders.

Batch modes
-----------
batch_size=1  : one track per step; uses _collate_single_track (no padding).
batch_size>1  : multi-track batches; uses _collate_variable_length_tracks.
                Variable-length tracks are padded to the longest in the batch.
                The batch dict gains a ``track_lengths`` key (B,) so _step
                can mask padded positions from the loss.

Species / max_tracks
--------------------
Set ``species`` to filter the Parquet to a single species (requires a
``species`` column; silently ignored if absent).
Set ``max_tracks`` to cap the training and validation sets to the N tracks
with the most observations — the first multi-track test uses 10.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import lightning as L
from torch.utils.data import DataLoader

from .dataset import PlantTrackStateDataset
from .transforms import GaussianStateTransformer


def _stratify_trays_by_coverage(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split tray_id into train/val/test so that coverage hours are stratified.
    Coverage per tray = max(time_since_germination_hours) in that tray.
    """
    if "time_since_germination_hours" not in df.columns:
        tray_coverage = df.groupby("tray_id").size()
    else:
        tray_coverage = df.groupby("tray_id")["time_since_germination_hours"].max()
    tray_ids = tray_coverage.index.tolist()
    if not tray_ids:
        return [], [], []

    # Sort by coverage and assign to bins so distribution is similar across splits
    sorted_trays = tray_coverage.sort_values().index.tolist()
    n = len(sorted_trays)
    n_train = max(1, int(n * train_frac))
    n_val = max(0, int(n * val_frac))
    n_test = max(0, n - n_train - n_val)

    # Interleave so low/medium/high coverage trays go to each split
    train_ids, val_ids, test_ids = [], [], []
    for i, tid in enumerate(sorted_trays):
        if i % 3 == 0 and len(train_ids) < n_train:
            train_ids.append(int(tid))
        elif i % 3 == 1 and len(val_ids) < n_val:
            val_ids.append(int(tid))
        elif len(test_ids) < n_test:
            test_ids.append(int(tid))
        elif len(train_ids) < n_train:
            train_ids.append(int(tid))
        elif len(val_ids) < n_val:
            val_ids.append(int(tid))
        else:
            test_ids.append(int(tid))

    return train_ids, val_ids, test_ids


def _select_top_n_tracks(
    df: pd.DataFrame,
    track_ids: list[int],
    n: int,
) -> list[int]:
    """Return the N track IDs with the most observations (most data-rich)."""
    counts = df[df["track_id"].isin(track_ids)].groupby("track_id").size()
    return counts.nlargest(n).index.tolist()


class PlantTrackDataModule(L.LightningDataModule):
    """
    Lightning-style DataModule: load Parquet, split by tray (stratified),
    fit transformer on train only, expose train/val/test dataloaders.

    Parameters
    ----------
    max_tracks  : Cap train/val sets to the N richest tracks.
                  None = use all available tracks (original behaviour).
    species     : If the Parquet has a ``species`` column, filter to this
                  value before any split.  Silently ignored if absent.
    """

    def __init__(
        self,
        parquet_path: Path | str,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        batch_size: int = 1,
        num_workers: int = 0,
        sigma_mode: str = "raw",
        max_tracks: Optional[int] = None,
        species: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.parquet_path = Path(parquet_path)
        self.train_frac   = train_frac
        self.val_frac     = val_frac
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.sigma_mode   = sigma_mode
        self.max_tracks   = max_tracks
        self.species      = species

        self._locked: bool = False
        self._transformer: Optional[GaussianStateTransformer] = None
        self._train_track_ids: Optional[list[int]] = None
        self._val_track_ids:   Optional[list[int]] = None
        self._test_track_ids:  Optional[list[int]] = None
        self._train_ds: Optional[PlantTrackStateDataset] = None
        self._val_ds:   Optional[PlantTrackStateDataset] = None
        self._test_ds:  Optional[PlantTrackStateDataset] = None

    def lock_setup(self) -> None:
        """Prevent subsequent calls to setup() from overwriting manually configured datasets."""
        self._locked = True

    def setup(self, stage: Optional[str] = None) -> None:
        if self._locked:
            return
        df = pd.read_parquet(self.parquet_path)
        if "valid_track" in df.columns:
            df = df[df["valid_track"]].copy()

        # Optional single-species filter
        if self.species is not None and "species" in df.columns:
            df = df[df["species"] == self.species].copy()
            if df.empty:
                raise ValueError(
                    f"No rows remain after filtering species={self.species!r}. "
                    f"Available: {df['species'].unique().tolist()}"
                )

        train_tray_ids, val_tray_ids, test_tray_ids = _stratify_trays_by_coverage(
            df, train_frac=self.train_frac, val_frac=self.val_frac
        )

        train_track_ids = df[df["tray_id"].isin(train_tray_ids)]["track_id"].unique().tolist()
        val_track_ids   = df[df["tray_id"].isin(val_tray_ids)]["track_id"].unique().tolist()
        test_track_ids  = df[df["tray_id"].isin(test_tray_ids)]["track_id"].unique().tolist()

        # Optionally cap to the most data-rich N tracks per split
        if self.max_tracks is not None:
            if len(train_track_ids) > self.max_tracks:
                train_track_ids = _select_top_n_tracks(df, train_track_ids, self.max_tracks)
            if val_track_ids and len(val_track_ids) > self.max_tracks:
                val_track_ids = _select_top_n_tracks(df, val_track_ids, self.max_tracks)

        train_df = df[df["track_id"].isin(train_track_ids)]

        transformer = GaussianStateTransformer(sigma_mode=self.sigma_mode)
        transformer.fit(train_df)

        self._transformer     = transformer
        self._train_track_ids = train_track_ids
        self._val_track_ids   = val_track_ids
        self._test_track_ids  = test_track_ids

        self._train_ds = PlantTrackStateDataset(
            self.parquet_path, transformer,
            track_ids=train_track_ids, valid_track_only=True,
        )
        self._val_ds = PlantTrackStateDataset(
            self.parquet_path, transformer,
            track_ids=val_track_ids, valid_track_only=True,
        )
        self._test_ds = PlantTrackStateDataset(
            self.parquet_path, transformer,
            track_ids=test_track_ids, valid_track_only=True,
        )

    @property
    def transformer(self) -> GaussianStateTransformer:
        if self._transformer is None:
            raise RuntimeError("Call setup() before accessing transformer")
        return self._transformer

    @transformer.setter
    def transformer(self, t: GaussianStateTransformer) -> None:
        self._transformer = t

    def _get_collate_fn(self):
        return (
            _collate_variable_length_tracks
            if self.batch_size > 1
            else _collate_single_track
        )

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            raise RuntimeError("Call setup() before requesting dataloaders")
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_ds is None:
            raise RuntimeError("Call setup() before requesting dataloaders")
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
        )

    def test_dataloader(self) -> DataLoader:
        if self._test_ds is None:
            raise RuntimeError("Call setup() before requesting dataloaders")
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
        )


# ── Collate functions ──────────────────────────────────────────────────────────

def _collate_single_track(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate a single-track batch (batch_size=1); just unwrap the list."""
    assert len(batch) == 1, "Use _collate_variable_length_tracks for batch_size > 1"
    return batch[0]


def _collate_variable_length_tracks(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Collate variable-length plant tracks into a padded batch.

    Pads all tracks to the length of the longest track in the batch:
      - states_5d : zero-padded                            (B, T_max, 5)
      - t_absolute: padded by repeating the last timestamp (B, T_max)
      - color_mask: padded with False                      (B, T_max) bool
    
    The returned ``track_lengths`` tensor holds the real (unpadded) length of
    each track so that _step can zero-out padded positions in the loss.

    Returns
    -------
    dict with keys:
      states_5d     (B, T_max, 5)  float
      t_absolute    (B, T_max)     float
      color_mask    (B, T_max)     bool
      track_lengths (B,)           long  — actual unpadded lengths
      track_ids     (B,)           long
    """
    states_list = [item["states_5d"] for item in batch]   # list of (T_i, 5)
    t_list      = [item["t_absolute"] for item in batch]  # list of (T_i,)
    mask_list   = [item["color_mask"] for item in batch]  # list of (T_i,) bool
    track_ids   = torch.stack([item["track_id"] for item in batch])

    lengths = torch.tensor([s.shape[0] for s in states_list], dtype=torch.long)
    T_max   = int(lengths.max().item())

    # States: pad with zeros → (B, T_max, 5)
    states_padded = pad_sequence(states_list, batch_first=True, padding_value=0.0)

    # t_absolute: extend shorter tracks with their last real timestamp so that
    # the time vector fed to odeint remains monotonically non-decreasing.
    t_padded = torch.zeros(len(batch), T_max, dtype=t_list[0].dtype)
    for i, (t, length) in enumerate(zip(t_list, lengths)):
        t_padded[i, :length] = t
        if length < T_max:
            t_padded[i, length:] = t[-1]

    # color_mask: pad with False (no physiology contribution from dummy frames)
    mask_padded = pad_sequence(
        [m.float() for m in mask_list], batch_first=True, padding_value=0.0
    ).bool()

    return {
        "states_5d":     states_padded,   # (B, T_max, 5)
        "t_absolute":    t_padded,        # (B, T_max)
        "color_mask":    mask_padded,     # (B, T_max) bool
        "track_lengths": lengths,         # (B,) long
        "track_ids":     track_ids,       # (B,) long
    }


# ── Z-Data coverage audit ─────────────────────────────────────────────────────

def audit_z_coverage(
    dm: "PlantTrackDataModule",
    label: str = "",
    n_tracks: int = 10,
) -> None:
    """
    Print Z (color_mask) coverage stats for each track in the training set.

    Call this after ``dm.setup()`` to verify that the selected tracks actually
    have valid Z observations before starting training.

    Parameters
    ----------
    dm      : a fitted PlantTrackDataModule (setup() already called)
    label   : optional string prefix for the header line
    n_tracks: cap — only print the first n_tracks to keep output manageable
    """
    ds = dm.train_dataloader().dataset
    header = f"[Z-Coverage Audit]{' ' + label if label else ''}"
    print(f"\n{header}  ({min(len(ds), n_tracks)} of {len(ds)} tracks)")
    any_empty = False
    for i in range(min(len(ds), n_tracks)):
        sample     = ds[i]
        n_total    = int(sample["color_mask"].shape[0])
        n_valid    = int(sample["color_mask"].sum().item())
        track_id   = int(sample["track_id"].item())
        frac       = n_valid / max(n_total, 1)
        flag       = "  ← NO VALID Z — physiology loss will be 0!" if n_valid == 0 else ""
        print(f"  Track {i:2d}  id={track_id:6d}  "
              f"{n_valid:4d}/{n_total:4d} valid Z frames  ({frac:5.1%}){flag}")
        if n_valid == 0:
            any_empty = True
    if any_empty:
        print("  WARNING: one or more tracks have zero Z observations.\n"
              "  Consider using --phys-loss-weight 0.0 or filtering those tracks.")
    print()
