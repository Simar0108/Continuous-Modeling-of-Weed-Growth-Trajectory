"""
Evaluation suite for comparing the Neural ODE against discrete baselines.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .train_baselines import BaselineLightning, build_baseline_datasets
from .train_multi import MultiTrackLightning


@dataclass
class ModelAdapter:
    name: str
    module: torch.nn.Module
    predict_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    n_context_frames: int
    param_count: int


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _masked_mse(
    pred_future: torch.Tensor,
    target_future: torch.Tensor,
    color_future: torch.Tensor,
    frame_mask: torch.Tensor | None = None,
) -> float:
    if frame_mask is None:
        frame_mask = torch.ones(
            pred_future.shape[0],
            dtype=torch.bool,
            device=pred_future.device,
        )

    values: list[torch.Tensor] = []
    size_err = (pred_future[frame_mask, :2] - target_future[frame_mask, :2]).pow(2).reshape(-1)
    if size_err.numel() > 0:
        values.append(size_err)

    z_mask = frame_mask & color_future
    z_err = (pred_future[z_mask, 2] - target_future[z_mask, 2]).pow(2)
    if z_err.numel() > 0:
        values.append(z_err)

    if not values:
        return float("nan")
    return float(torch.cat(values).mean().item())


def _shape_correlation(pred_future: torch.Tensor, target_future: torch.Tensor) -> float:
    preds = pred_future.detach().cpu().numpy()
    target = target_future.detach().cpu().numpy()
    corrs = [
        _pearsonr(preds[:, 0], target[:, 0]),
        _pearsonr(preds[:, 1], target[:, 1]),
    ]
    corrs = [c for c in corrs if np.isfinite(c)]
    return float(np.mean(corrs)) if corrs else float("nan")


def _mean_squared_jerk(pred_future: torch.Tensor, t_future: torch.Tensor) -> float:
    if pred_future.shape[0] < 4:
        return float("nan")
    values = pred_future.detach().cpu().numpy()
    times = t_future.detach().cpu().numpy()

    jerks: list[float] = []
    for channel in range(values.shape[1]):
        vel = np.gradient(values[:, channel], times, edge_order=1)
        acc = np.gradient(vel, times, edge_order=1)
        jerk = np.gradient(acc, times, edge_order=1)
        jerks.append(float(np.mean(jerk ** 2)))
    return float(np.mean(jerks))


def _build_context_only_query(
    states_5d: torch.Tensor,
    t_absolute: torch.Tensor,
    *,
    context_indices: torch.Tensor,
    prediction_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build an inference query with real context frames and placeholder future states.

    This guarantees that evaluation never leaks future ground-truth states into the
    discrete baselines: only the first K context frames carry observations.
    """
    context_states = states_5d[context_indices]
    future_zeros = torch.zeros(
        prediction_indices.numel(),
        states_5d.shape[-1],
        dtype=states_5d.dtype,
        device=states_5d.device,
    )
    query_states = torch.cat([context_states, future_zeros], dim=0)
    query_times = torch.cat([t_absolute[context_indices], t_absolute[prediction_indices]], dim=0)
    return query_states, query_times


def _irregular_drop_mask(length: int, track_id: int, seed: int, drop_frac: float) -> torch.Tensor:
    if length <= 0:
        return torch.zeros(length, dtype=torch.bool)
    rng = np.random.default_rng(seed + track_id)
    n_drop = max(1, int(round(length * drop_frac)))
    chosen = rng.choice(length, size=min(n_drop, length), replace=False)
    mask = np.zeros(length, dtype=bool)
    mask[chosen] = True
    return torch.from_numpy(mask)


def _load_ode_adapter(checkpoint: Path, device: torch.device) -> ModelAdapter:
    from .overfit_test import OverfitLightning

    errors: list[str] = []
    module = None
    for cls in (OverfitLightning, MultiTrackLightning):
        try:
            module = cls.load_from_checkpoint(str(checkpoint), map_location=device, strict=False)
            break
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")
    if module is None:
        raise RuntimeError(f"Failed to load ODE checkpoint:\n" + "; ".join(errors))
    module.eval().to(device)
    n_context_frames = int(getattr(module, "_n_context_frames", 3))

    def _predict(states_5d: torch.Tensor, t_absolute: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            pred = module.model(states_5d.unsqueeze(0), t_absolute, return_aux=False)
        pred = pred.permute(1, 0, 2).squeeze(0)
        return pred[n_context_frames:, 2:5]

    return ModelAdapter(
        name="ode",
        module=module,
        predict_fn=_predict,
        n_context_frames=n_context_frames,
        param_count=_count_parameters(module.model),
    )


def _load_baseline_adapter(name: str, checkpoint: Path, device: torch.device) -> ModelAdapter:
    module = BaselineLightning.load_from_checkpoint(
        str(checkpoint), map_location=device, strict=False
    )
    module.eval().to(device)
    n_context_frames = int(getattr(module, "n_context_frames", 3))

    def _predict(states_5d: torch.Tensor, t_absolute: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            pred = module.model.autoregressive_rollout(
                states_5d.unsqueeze(0),
                t_absolute,
                torch.tensor([states_5d.shape[0]], device=states_5d.device),
            )
        return pred.squeeze(0)

    return ModelAdapter(
        name=name,
        module=module,
        predict_fn=_predict,
        n_context_frames=n_context_frames,
        param_count=_count_parameters(module.model),
    )


def _evaluate_model(
    adapter: ModelAdapter,
    dataset,
    *,
    device: torch.device,
    seed: int,
    drop_frac: float,
) -> dict[str, float]:
    shape_corrs: list[float] = []
    extrap_mses: list[float] = []
    irregular_mses: list[float] = []
    smoothness_vals: list[float] = []
    inference_times_ms: list[float] = []

    for sample in dataset:
        states_5d = sample["states_5d"].to(device)
        t_absolute = sample["t_absolute"].to(device)
        color_mask = sample["color_mask"].to(device)
        track_id = int(sample["track_id"].item())

        k = adapter.n_context_frames
        if states_5d.shape[0] <= k:
            continue

        dense_context = torch.arange(k, device=device)
        dense_future = torch.arange(k, states_5d.shape[0], device=device)
        dense_query_states, dense_query_times = _build_context_only_query(
            states_5d,
            t_absolute,
            context_indices=dense_context,
            prediction_indices=dense_future,
        )
        target_future = states_5d[dense_future, 2:5]
        color_future = color_mask[dense_future]
        t_future = t_absolute[dense_future]

        _sync_if_needed(device)
        start = time.perf_counter()
        pred_future = adapter.predict_fn(dense_query_states, dense_query_times)
        _sync_if_needed(device)
        inference_times_ms.append((time.perf_counter() - start) * 1000.0)

        shape_corrs.append(_shape_correlation(pred_future, target_future))

        total_len = states_5d.shape[0]
        tail_start = max(k, int(np.floor(total_len * 0.7)))
        tail_mask = torch.arange(target_future.shape[0], device=device) >= (tail_start - k)
        extrap_mses.append(_masked_mse(pred_future, target_future, color_future, tail_mask))

        keep_mask = ~_irregular_drop_mask(states_5d.shape[0], track_id, seed, drop_frac).to(device)
        keep_mask[0] = True
        retained_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
        if retained_indices.numel() <= k:
            retained_indices = torch.arange(states_5d.shape[0], device=device)

        irregular_context = retained_indices[:k]
        irregular_future = torch.arange(irregular_context[-1].item() + 1, states_5d.shape[0], device=device)
        if irregular_future.numel() == 0:
            irregular_mses.append(float("nan"))
            smoothness_vals.append(_mean_squared_jerk(pred_future, t_future))
            continue

        irregular_query_states, irregular_query_times = _build_context_only_query(
            states_5d,
            t_absolute,
            context_indices=irregular_context,
            prediction_indices=irregular_future,
        )
        irregular_pred = adapter.predict_fn(irregular_query_states, irregular_query_times)
        irregular_target = states_5d[irregular_future, 2:5]
        irregular_color = color_mask[irregular_future]
        dropped_future_mask = ~keep_mask[irregular_future]
        irregular_mses.append(
            _masked_mse(irregular_pred, irregular_target, irregular_color, dropped_future_mask)
        )

        smoothness_vals.append(_mean_squared_jerk(pred_future, t_future))

    def _finite_mean(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else float("nan")

    return {
        "pearson_shape_corr": _finite_mean(shape_corrs),
        "extrapolation_mse": _finite_mean(extrap_mses),
        "irregular_sampling_mse": _finite_mean(irregular_mses),
        "trajectory_smoothness_msj": _finite_mean(smoothness_vals),
        "inference_ms_per_track": _finite_mean(inference_times_ms),
        "param_count": float(adapter.param_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ODE and discrete baseline checkpoints")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--sigma-mode", choices=("raw", "z_score"), default="z_score")
    parser.add_argument("--max-tracks", type=int, default=100)
    parser.add_argument("--species", type=str, default=None)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--split", choices=("val", "test"), default="test")

    parser.add_argument("--ode-checkpoint", type=Path, required=True)
    parser.add_argument("--lstm-checkpoint", type=Path, required=True)
    parser.add_argument("--gru-checkpoint", type=Path, required=True)
    parser.add_argument("--transformer-checkpoint", type=Path, required=True)

    parser.add_argument("--drop-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    parquet_path = args.parquet.expanduser()
    if not parquet_path.is_absolute():
        parquet_path = Path(__file__).parent.parent / parquet_path
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Parquet not found: {args.parquet!r}\n  Resolved -> {parquet_path.resolve()}"
        )

    _, _, val_ds, test_ds = build_baseline_datasets(
        parquet_path,
        sigma_mode=args.sigma_mode,
        max_tracks=args.max_tracks,
        species=args.species,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    dataset = test_ds if args.split == "test" else val_ds
    device = _resolve_device(args.device)

    adapters = [
        _load_ode_adapter(args.ode_checkpoint.expanduser(), device),
        _load_baseline_adapter("lstm", args.lstm_checkpoint.expanduser(), device),
        _load_baseline_adapter("gru", args.gru_checkpoint.expanduser(), device),
        _load_baseline_adapter("transformer", args.transformer_checkpoint.expanduser(), device),
    ]

    rows = []
    for adapter in adapters:
        metrics = _evaluate_model(
            adapter,
            dataset,
            device=device,
            seed=args.seed,
            drop_frac=args.drop_frac,
        )
        rows.append({"model": adapter.name, **metrics})

    df = pd.DataFrame(rows).set_index("model").sort_index()
    print(f"\n[compare_models] Split={args.split}  n_tracks={len(dataset)}")
    print(df.to_string(float_format=lambda v: f"{v:.6g}"))

    if args.output is not None:
        output_path = args.output.expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        print(f"\n[compare_models] Wrote metrics -> {output_path}")


if __name__ == "__main__":
    main()
