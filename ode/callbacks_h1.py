"""H1 selection-metric callbacks. Do not edit training_loop hybrid_loss."""

from __future__ import annotations

import torch
from lightning.pytorch.callbacks import Callback


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _track_mse_from_logs(logs: dict) -> torch.Tensor:
    for key, value in logs.items():
        if str(key).endswith("_track_mse_mean"):
            if torch.is_tensor(value):
                return value.detach()
            return torch.tensor(float(value))
    raise KeyError(f"no *_track_mse_mean in {list(logs)}")


class FixedHorizonValCallback(Callback):
    """Re-score val at horizon 1.0 and log ``val_full_horizon_mse``.

    Training still uses the ramping curriculum. Checkpoint selection should
    monitor this metric so early short-horizon val cannot beat later full
    trajectories.
    """

    def __init__(self, every_n_epochs: int = 1, monitor_key: str = "val_full_horizon_mse"):
        super().__init__()
        self.every_n_epochs = max(int(every_n_epochs), 1)
        self.monitor_key = monitor_key

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        if int(trainer.current_epoch) % self.every_n_epochs != 0:
            return
        datamodule = trainer.datamodule
        if datamodule is None:
            return
        saved = (
            float(pl_module.horizon_start_frac),
            int(pl_module.horizon_ramp_start),
            int(pl_module.horizon_ramp_end),
        )
        pl_module.horizon_start_frac = 1.0
        pl_module.horizon_ramp_start = 0
        pl_module.horizon_ramp_end = 0
        device = pl_module.device
        mses: list[torch.Tensor] = []
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            for batch in datamodule.val_dataloader():
                batch = _move_batch(batch, device)
                _loss, logs, _bs = pl_module._step(batch, "valfull")
                mses.append(_track_mse_from_logs(logs).to(device))
        pl_module.horizon_start_frac, pl_module.horizon_ramp_start, pl_module.horizon_ramp_end = saved
        if was_training:
            pl_module.train()
        if not mses:
            return
        metric = torch.stack(mses).mean()
        pl_module.log(
            self.monitor_key, metric, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        pl_module.log(
            "val_full_horizon_frac", torch.tensor(1.0, device=device),
            on_step=False, on_epoch=True, sync_dist=True,
        )


class BestEpochGateCallback(Callback):
    """After fit, classify best-val epoch < 20 as TRAINING FAILURE."""

    def __init__(self, checkpoint_cb, min_epoch: int = 20):
        super().__init__()
        self.checkpoint_cb = checkpoint_cb
        self.min_epoch = int(min_epoch)
        self.best_epoch = -1
        self.training_failure = True

    def on_fit_end(self, trainer, pl_module) -> None:
        path = getattr(self.checkpoint_cb, "best_model_path", "") or ""
        epoch = int(trainer.current_epoch)
        if path:
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                epoch = int(ckpt.get("epoch", epoch))
            except TypeError:
                ckpt = torch.load(path, map_location="cpu")
                epoch = int(ckpt.get("epoch", epoch))
            except Exception:
                pass
        self.best_epoch = epoch
        self.training_failure = epoch < self.min_epoch
        tag = "TRAINING FAILURE" if self.training_failure else "converged"
        print(
            f"[h1_gate] best_epoch={epoch} monitor={getattr(self.checkpoint_cb, 'monitor', '?')} "
            f"class={tag} (fail if epoch < {self.min_epoch})"
        )
        try:
            pl_module.log("best_epoch_at_fit_end", float(epoch), on_epoch=True)
        except Exception:
            pass
        if getattr(trainer, "logger", None) is not None:
            try:
                trainer.logger.log_metrics({
                    "best_epoch": float(epoch),
                    "training_failure": float(self.training_failure),
                }, step=trainer.global_step)
            except Exception:
                pass
