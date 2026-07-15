"""Small deterministic PyTorch sequence models for offline research.

The dataset stores one 2-D feature matrix and a vector of valid end points; it
does not materialize the much larger ``samples x lookback x features`` tensor.
Training never makes a random validation split.  A chronological validation
dataset must be supplied explicitly if early stopping is requested.
"""

from __future__ import annotations

import copy
import os
import random
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DeviceLike = Union[str, torch.device, None]


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and Torch for repeatable CPU/GPU research runs."""

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # CUDA matrix multiplication/RNN kernels require this workspace contract
    # for bitwise deterministic execution.  It must be set before a CUDA
    # context is first used, which the estimator constructors guarantee.
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # Memory-efficient/flash attention can be nondeterministic on CUDA.
        # The math implementation is appropriate for these deliberately small
        # research models and preserves seeded repeatability.
        if hasattr(torch.backends, "cuda"):
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)


def resolve_device(device: DeviceLike = None) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


class LazySequenceDataset(Dataset):
    """Create fixed-length sequences lazily from a sorted long feature matrix.

    Parameters
    ----------
    features:
        ``[rows, features]`` values sorted by group then time.
    targets:
        Optional row-aligned scalar targets.  The target at a sequence's final
        row is returned.
    groups:
        Optional row-aligned symbol/entity identifiers.  A sequence never
        crosses a group boundary.  Each group must occupy one contiguous block.
    sequence_length:
        Number of rows in each sequence.
    allowed_endpoint_mask / allowed_endpoint_indices:
        Optional explicit row-level whitelist for sequence endpoints.  Lookback
        rows may precede the selected interval, but every returned target and
        prediction is anchored to an allowed close(t) row.  Supplying both is
        rejected.
    """

    def __init__(
        self,
        features: Union[np.ndarray, torch.Tensor, Sequence[Sequence[float]]],
        targets: Optional[Union[np.ndarray, torch.Tensor, Sequence[float]]] = None,
        *,
        groups: Optional[Sequence[object]] = None,
        sequence_length: int = 60,
        fill_value: float = 0.0,
        allowed_endpoint_mask: Optional[Sequence[bool]] = None,
        allowed_endpoint_indices: Optional[Sequence[int]] = None,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        if feature_tensor.ndim != 2:
            raise ValueError(
                f"features must have shape [rows, features]; got {feature_tensor.shape}"
            )
        if feature_tensor.shape[0] == 0 or feature_tensor.shape[1] == 0:
            raise ValueError("features must be non-empty")
        self._features = feature_tensor.contiguous()
        self.sequence_length = int(sequence_length)
        self.fill_value = float(fill_value)

        if targets is None:
            self._targets = None
        else:
            target_tensor = torch.as_tensor(targets, dtype=torch.float32).reshape(-1)
            if target_tensor.shape[0] != feature_tensor.shape[0]:
                raise ValueError("targets must have one value per feature row")
            self._targets = target_tensor.contiguous()

        row_count = feature_tensor.shape[0]
        group_values = (
            np.zeros(row_count, dtype=np.int8)
            if groups is None
            else np.asarray(groups)
        )
        if group_values.ndim != 1 or group_values.shape[0] != row_count:
            raise ValueError("groups must be one-dimensional and row-aligned")

        boundaries = np.flatnonzero(group_values[1:] != group_values[:-1]) + 1
        starts = np.r_[0, boundaries]
        stops = np.r_[boundaries, row_count]

        # Reappearing groups almost always mean the input was not sorted by
        # symbol/time.  Reject it instead of silently creating invalid samples.
        seen: set[object] = set()
        for start in starts:
            key = group_values[int(start)].item() if hasattr(group_values[int(start)], "item") else group_values[int(start)]
            try:
                if key in seen:
                    raise ValueError("each group must occupy one contiguous block")
                seen.add(key)
            except TypeError as exc:
                raise ValueError("group identifiers must be hashable") from exc

        if allowed_endpoint_mask is not None and allowed_endpoint_indices is not None:
            raise ValueError(
                "allowed_endpoint_mask and allowed_endpoint_indices are mutually exclusive"
            )

        endpoint_parts = []
        for start, stop in zip(starts, stops):
            first_end = int(start) + self.sequence_length - 1
            if first_end < int(stop):
                endpoint_parts.append(np.arange(first_end, int(stop), dtype=np.int64))
        natural_endpoints = (
            np.concatenate(endpoint_parts)
            if endpoint_parts
            else np.empty(0, dtype=np.int64)
        )
        if allowed_endpoint_mask is not None:
            endpoint_mask = np.asarray(allowed_endpoint_mask)
            if endpoint_mask.ndim != 1 or endpoint_mask.shape[0] != row_count:
                raise ValueError(
                    "allowed_endpoint_mask must be one-dimensional and row-aligned"
                )
            if not np.issubdtype(endpoint_mask.dtype, np.bool_):
                raise ValueError("allowed_endpoint_mask must contain booleans")
            natural_endpoints = natural_endpoints[endpoint_mask[natural_endpoints]]
        elif allowed_endpoint_indices is not None:
            allowed = np.asarray(allowed_endpoint_indices)
            if allowed.ndim != 1:
                raise ValueError("allowed_endpoint_indices must be a 1-D integer vector")
            if allowed.size == 0:
                allowed = np.empty(0, dtype=np.int64)
            elif not np.issubdtype(allowed.dtype, np.integer):
                raise ValueError("allowed_endpoint_indices must be a 1-D integer vector")
            else:
                allowed = allowed.astype(np.int64, copy=False)
            if allowed.size and (allowed.min() < 0 or allowed.max() >= row_count):
                raise ValueError("allowed endpoint index is outside the feature rows")
            if np.unique(allowed).size != allowed.size:
                raise ValueError("allowed_endpoint_indices must be unique")
            natural_endpoints = np.intersect1d(
                natural_endpoints,
                allowed,
                assume_unique=True,
            )
        self.end_indices = natural_endpoints

    @property
    def has_targets(self) -> bool:
        return self._targets is not None

    @property
    def feature_count(self) -> int:
        return int(self._features.shape[1])

    @property
    def storage_shape(self) -> tuple[int, int]:
        """Shape of the sole feature storage, useful for memory assertions."""

        return tuple(self._features.shape)  # type: ignore[return-value]

    def __len__(self) -> int:
        return int(self.end_indices.size)

    def __getitem__(self, index: int):
        endpoint = int(self.end_indices[index])
        start = endpoint - self.sequence_length + 1
        sequence = self._features[start : endpoint + 1]
        if not torch.isfinite(sequence).all():
            sequence = torch.nan_to_num(
                sequence,
                nan=self.fill_value,
                posinf=self.fill_value,
                neginf=self.fill_value,
            )
        if self._targets is None:
            return sequence
        return sequence, self._targets[endpoint]


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _enforce_parameter_limit(model: nn.Module, max_parameters: int) -> None:
    count = count_trainable_parameters(model)
    if max_parameters < 1:
        raise ValueError("max_parameters must be positive")
    if count > max_parameters:
        raise ValueError(
            f"model has {count:,} trainable parameters, exceeding limit "
            f"{max_parameters:,}"
        )
    model.parameter_count = count  # type: ignore[attr-defined]
    model.max_parameters = int(max_parameters)  # type: ignore[attr-defined]


class SmallGRU(nn.Module):
    """Compact last-state GRU regressor."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_dim: int = 1,
        max_parameters: int = 500_000,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or num_layers < 1 or output_dim < 1:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        effective_dropout = float(dropout) if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=effective_dropout,
        )
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.head = nn.Linear(int(hidden_dim), int(output_dim))
        self.output_dim = int(output_dim)
        _enforce_parameter_limit(self, max_parameters)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("GRU input must have shape [batch, sequence, features]")
        outputs, _ = self.gru(inputs)
        prediction = self.head(self.norm(outputs[:, -1, :]))
        return prediction.squeeze(-1) if self.output_dim == 1 else prediction


class SmallTransformer(nn.Module):
    """Compact encoder-only Transformer regressor with learned positions."""

    def __init__(
        self,
        input_dim: int,
        *,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.0,
        max_sequence_length: int = 120,
        output_dim: int = 1,
        max_parameters: int = 500_000,
    ) -> None:
        super().__init__()
        if min(input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim) < 1:
            raise ValueError("model dimensions must be positive")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_projection = nn.Linear(int(input_dim), int(d_model))
        self.position = nn.Parameter(
            torch.zeros(1, int(max_sequence_length), int(d_model))
        )
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(num_layers),
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(int(d_model))
        self.head = nn.Linear(int(d_model), int(output_dim))
        self.max_sequence_length = int(max_sequence_length)
        self.output_dim = int(output_dim)
        _enforce_parameter_limit(self, max_parameters)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                "Transformer input must have shape [batch, sequence, features]"
            )
        sequence_length = inputs.shape[1]
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds configured maximum "
                f"{self.max_sequence_length}"
            )
        hidden = self.input_projection(inputs)
        hidden = hidden + self.position[:, :sequence_length, :]
        hidden = self.encoder(hidden)
        prediction = self.head(self.norm(hidden[:, -1, :]))
        return prediction.squeeze(-1) if self.output_dim == 1 else prediction


def _worker_seed(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TorchSequenceRegressor:
    """Estimator-like deterministic fit/predict wrapper around a Torch model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        seed: int = 42,
        device: DeviceLike = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip: Optional[float] = 1.0,
    ) -> None:
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("invalid optimizer parameter")
        self.seed = int(seed)
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.gradient_clip = gradient_clip
        self.history_: list[dict[str, float]] = []
        self.is_fitted_ = False

    def _loader(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
    ) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=shuffle,
            num_workers=int(num_workers),
            generator=generator,
            worker_init_fn=_worker_seed if num_workers else None,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

    def fit(
        self,
        dataset: LazySequenceDataset,
        *,
        epochs: int = 5,
        batch_size: int = 256,
        validation_dataset: Optional[LazySequenceDataset] = None,
        patience: Optional[int] = None,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> "TorchSequenceRegressor":
        if len(dataset) == 0 or not dataset.has_targets:
            raise ValueError("training dataset must contain sequences and targets")
        if epochs < 1 or batch_size < 1 or num_workers < 0:
            raise ValueError("invalid training-loop parameter")
        if validation_dataset is not None and (
            len(validation_dataset) == 0 or not validation_dataset.has_targets
        ):
            raise ValueError("validation dataset must contain sequences and targets")
        if patience is not None:
            if validation_dataset is None:
                raise ValueError(
                    "patience requires an explicitly supplied chronological "
                    "validation_dataset"
                )
            if patience < 1:
                raise ValueError("patience must be positive")

        seed_everything(self.seed)
        train_loader = self._loader(
            dataset,
            batch_size=batch_size,
            shuffle=bool(shuffle),
            num_workers=num_workers,
        )
        validation_loader = (
            self._loader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            if validation_dataset is not None
            else None
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()
        self.history_ = []
        best_loss = float("inf")
        best_state = None
        stale_epochs = 0

        for epoch in range(int(epochs)):
            self.model.train()
            total_loss = 0.0
            total_rows = 0
            for features, targets in train_loader:
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                predictions = self.model(features)
                loss = criterion(predictions.reshape_as(targets), targets)
                loss.backward()
                if self.gradient_clip is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), float(self.gradient_clip)
                    )
                optimizer.step()
                rows = int(targets.numel())
                total_loss += float(loss.detach().cpu()) * rows
                total_rows += rows

            record = {
                "epoch": float(epoch + 1),
                "train_loss": total_loss / max(total_rows, 1),
            }
            if validation_loader is not None:
                validation_loss = self._loss(validation_loader, criterion)
                record["validation_loss"] = validation_loss
                if validation_loss < best_loss - 1e-12:
                    best_loss = validation_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            self.history_.append(record)
            if patience is not None and stale_epochs >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.is_fitted_ = True
        return self

    def _loss(self, loader: DataLoader, criterion: nn.Module) -> float:
        self.model.eval()
        total_loss = 0.0
        total_rows = 0
        with torch.no_grad():
            for features, targets in loader:
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                predictions = self.model(features)
                loss = criterion(predictions.reshape_as(targets), targets)
                rows = int(targets.numel())
                total_loss += float(loss.detach().cpu()) * rows
                total_rows += rows
        return total_loss / max(total_rows, 1)

    def predict(
        self,
        dataset: LazySequenceDataset,
        *,
        batch_size: int = 512,
        num_workers: int = 0,
    ) -> np.ndarray:
        if len(dataset) == 0:
            return np.empty(0, dtype=np.float32)
        loader = self._loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self.model.eval()
        batches = []
        with torch.no_grad():
            for batch in loader:
                features = batch[0] if isinstance(batch, (tuple, list)) else batch
                prediction = self.model(features.to(self.device, non_blocking=True))
                batches.append(prediction.detach().cpu().numpy())
        return np.concatenate(batches, axis=0)


class GRUSequenceRegressor(TorchSequenceRegressor):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        max_parameters: int = 500_000,
        seed: int = 42,
        device: DeviceLike = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip: Optional[float] = 1.0,
    ) -> None:
        seed_everything(seed)
        model = SmallGRU(
            input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            max_parameters=max_parameters,
        )
        super().__init__(
            model,
            seed=seed,
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
        )


class TransformerSequenceRegressor(TorchSequenceRegressor):
    def __init__(
        self,
        input_dim: int,
        *,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.0,
        max_sequence_length: int = 120,
        max_parameters: int = 500_000,
        seed: int = 42,
        device: DeviceLike = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip: Optional[float] = 1.0,
    ) -> None:
        seed_everything(seed)
        model = SmallTransformer(
            input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_sequence_length=max_sequence_length,
            max_parameters=max_parameters,
        )
        super().__init__(
            model,
            seed=seed,
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
        )


__all__ = [
    "GRUSequenceRegressor",
    "LazySequenceDataset",
    "SmallGRU",
    "SmallTransformer",
    "TorchSequenceRegressor",
    "TransformerSequenceRegressor",
    "count_trainable_parameters",
    "resolve_device",
    "seed_everything",
]
