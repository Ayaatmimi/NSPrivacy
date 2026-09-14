"""Candidate NSPrivacy DP-SGD pipeline with explicit RDP accounting.

The accountant covers only the private training updates performed here.
Dataset-specific preprocessing, tuning, and evaluation releases are separate.
"""

from __future__ import annotations

import copy
import json
import math
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    input_dim: int
    num_classes: int
    target_epsilon: float
    target_delta: float
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    erase_sigma: float = 0.05
    lambda_sparse: float = 1e-4
    lambda_design: float = 1e-4
    warm_epochs: int = 10
    public_seed: int = 0
    private_seed: Optional[int] = None
    class_weights: Optional[Sequence[float]] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def validate(self) -> None:
        if self.input_dim < 1 or self.num_classes < 2:
            raise ValueError("input_dim and num_classes are invalid")
        if self.target_epsilon <= 0 or not 0 < self.target_delta < 1:
            raise ValueError("epsilon must be positive and delta must be in (0, 1)")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.max_grad_norm <= 0 or self.erase_sigma < 0:
            raise ValueError("clipping norm must be positive and erase_sigma nonnegative")
        if not 0 <= self.warm_epochs <= self.epochs:
            raise ValueError("warm_epochs must be between zero and epochs")
        if self.class_weights is not None:
            if len(self.class_weights) != self.num_classes:
                raise ValueError("class_weights must contain one value per class")
            if any(float(w) <= 0 for w in self.class_weights):
                raise ValueError("class_weights must be strictly positive")


class NSPrivacyNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, erase_sigma: float):
        super().__init__()
        self.num_classes = num_classes
        self.erase_sigma = float(erase_sigma)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.mask_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Sigmoid(),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        apply_non_retention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        mask = self.mask_net(z)
        z_masked = z * mask

        if apply_non_retention:
            if y is None:
                raise ValueError("labels are required for non-retention noise")
            reference_logits = self.classifier(z_masked)
            probabilities = reference_logits.softmax(dim=-1)
            classes = torch.arange(self.num_classes, device=y.device)
            targets = (y.unsqueeze(-1) == classes).to(z_masked.dtype)
            sensitivity = (probabilities - targets) @ self.classifier.weight
            scale = self.erase_sigma * sensitivity.abs().detach()
            z_masked = z_masked + torch.randn_like(z_masked) * scale

        logits = self.classifier(z_masked)
        return logits, z, mask


def _single_record_loss(
    params: Dict[str, torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    model: NSPrivacyNet,
    x: torch.Tensor,
    y: torch.Tensor,
    class_weights: torch.Tensor,
    lambda_sparse: float,
    lambda_design: float,
) -> torch.Tensor:
    logits, z, _ = functional_call(
        model,
        (params, buffers),
        (x.unsqueeze(0), y.unsqueeze(0), True),
        strict=False,
    )
    prediction_loss = F.cross_entropy(
        logits,
        y.unsqueeze(0),
        weight=class_weights,
        reduction="mean",
    )
    sparse_weight = params.get("mask_net.2.weight")
    if sparse_weight is None:
        sparse_penalty = prediction_loss.new_zeros(())
    else:
        sparse_penalty = sparse_weight.abs().sum()
    design_penalty = z.square().sum()
    return (
        prediction_loss
        + float(lambda_sparse) * sparse_penalty
        + float(lambda_design) * design_penalty
    )


_single_record_grad = grad(_single_record_loss)


def _per_record_gradients(
    model: NSPrivacyNet,
    x: torch.Tensor,
    y: torch.Tensor,
    class_weights: torch.Tensor,
    lambda_sparse: float,
    lambda_design: float,
) -> Dict[str, torch.Tensor]:
    params = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    buffers = dict(model.named_buffers())
    return vmap(
        _single_record_grad,
        in_dims=(None, None, None, 0, 0, None, None, None),
        randomness="different",
    )(
        params,
        buffers,
        model,
        x,
        y,
        class_weights,
        lambda_sparse,
        lambda_design,
    )


def _set_private_gradients(
    model: NSPrivacyNet,
    per_record: Optional[Dict[str, torch.Tensor]],
    expected_batch_size: int,
    max_grad_norm: float,
    noise_multiplier: float,
    noise_generator: torch.Generator,
) -> None:
    active = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    if per_record:
        first = next(iter(per_record.values()))
        batch_size = first.shape[0]
        squared_norm = torch.zeros(batch_size, device=first.device)
        for record_grad in per_record.values():
            squared_norm.add_(record_grad.flatten(1).square().sum(dim=1))
        norm = squared_norm.sqrt()
        factors = (max_grad_norm / norm.clamp_min(1e-12)).clamp(max=1.0)
    else:
        factors = None

    standard_deviation = noise_multiplier * max_grad_norm
    for name, parameter in active.items():
        if per_record:
            record_grad = per_record[name]
            view_shape = (record_grad.shape[0],) + (1,) * (record_grad.ndim - 1)
            clipped_sum = (record_grad * factors.view(view_shape)).sum(dim=0)
        else:
            clipped_sum = torch.zeros_like(parameter)

        noise = torch.randn(
            parameter.shape,
            generator=noise_generator,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        parameter.grad = (
            clipped_sum + standard_deviation * noise
        ) / float(expected_batch_size)


def _freeze_mask(model: NSPrivacyNet) -> None:
    for parameter in model.mask_net.parameters():
        parameter.requires_grad_(False)


def _private_generator(device: torch.device, seed: Optional[int]) -> torch.Generator:
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(secrets.randbits(63) if seed is None else int(seed))
    return generator


def train_private(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    config: TrainConfig,
) -> Tuple[NSPrivacyNet, dict]:
    config.validate()
    if x_train.ndim != 2 or y_train.ndim != 1:
        raise ValueError("x_train must be 2-D and y_train must be 1-D")
    if len(x_train) != len(y_train) or len(x_train) == 0:
        raise ValueError("training arrays have incompatible sizes")
    if x_train.shape[1] != config.input_dim:
        raise ValueError("input_dim does not match x_train")
    if int(y_train.min()) < 0 or int(y_train.max()) >= config.num_classes:
        raise ValueError("labels must be in 0,...,num_classes-1")

    torch.manual_seed(config.public_seed)
    device = torch.device(config.device)
    model = NSPrivacyNet(
        config.input_dim,
        config.num_classes,
        config.erase_sigma,
    ).to(device)

    dataset = TensorDataset(x_train.cpu().float(), y_train.cpu().long())
    base_loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=False,
        drop_last=False,
    )
    sample_rate = 1.0 / len(base_loader)
    sampling_generator = _private_generator(torch.device("cpu"), config.private_seed)
    private_loader = DPDataLoader.from_data_loader(
        base_loader,
        generator=sampling_generator,
    )
    expected_batch_size = max(1, int(round(len(dataset) * sample_rate)))
    planned_steps = config.epochs * len(private_loader)

    noise_multiplier = get_noise_multiplier(
        target_epsilon=config.target_epsilon,
        target_delta=config.target_delta,
        sample_rate=sample_rate,
        steps=planned_steps,
        accountant="rdp",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.epochs),
    )
    accountant = RDPAccountant()
    noise_generator = _private_generator(device, config.private_seed)

    if config.class_weights is None:
        class_weights = torch.ones(config.num_classes, device=device)
    else:
        class_weights = torch.tensor(
            config.class_weights,
            dtype=torch.float32,
            device=device,
        )

    completed_steps = 0
    completed_epochs = 0
    stopped_for_budget = False

    model.train()
    for epoch in range(config.epochs):
        if epoch == config.warm_epochs:
            _freeze_mask(model)

        if config.warm_epochs == 0:
            design_weight = config.lambda_design
        else:
            design_weight = config.lambda_design * min(
                1.0,
                float(epoch + 1) / float(config.warm_epochs),
            )

        for x_batch, y_batch in private_loader:
            projected = copy.deepcopy(accountant)
            projected.step(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
            )
            if projected.get_epsilon(config.target_delta) > config.target_epsilon + 1e-6:
                stopped_for_budget = True
                break

            optimizer.zero_grad(set_to_none=True)
            if len(x_batch) == 0:
                per_record = None
            else:
                x_batch = x_batch.to(device=device, dtype=torch.float32)
                y_batch = y_batch.to(device=device, dtype=torch.long)
                per_record = _per_record_gradients(
                    model,
                    x_batch,
                    y_batch,
                    class_weights,
                    config.lambda_sparse,
                    design_weight,
                )

            _set_private_gradients(
                model,
                per_record,
                expected_batch_size,
                config.max_grad_norm,
                noise_multiplier,
                noise_generator,
            )
            optimizer.step()
            accountant.step(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
            )
            completed_steps += 1

        completed_epochs = epoch + 1
        scheduler.step()
        if stopped_for_budget:
            break

    epsilon = (
        accountant.get_epsilon(config.target_delta)
        if completed_steps
        else 0.0
    )
    audit = {
        "implementation_status": "candidate",
        "target_epsilon": config.target_epsilon,
        "achieved_epsilon": float(epsilon),
        "delta": config.target_delta,
        "noise_multiplier": float(noise_multiplier),
        "sample_rate": float(sample_rate),
        "max_grad_norm": config.max_grad_norm,
        "planned_steps": planned_steps,
        "completed_steps": completed_steps,
        "completed_epochs": completed_epochs,
        "stopped_before_budget_exceedance": stopped_for_budget,
        "mask_warm_epochs": config.warm_epochs,
        "private_rng_mode": (
            "operating-system-seeded"
            if config.private_seed is None
            else "fixed-debug-seed"
        ),
    }
    return model, audit


@torch.no_grad()
def evaluate(
    model: NSPrivacyNet,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 1024,
) -> dict:
    device = next(model.parameters()).device
    loader = DataLoader(
        TensorDataset(x.cpu().float(), y.cpu().long()),
        batch_size=batch_size,
        shuffle=False,
    )
    probabilities = []
    targets = []
    model.eval()
    for x_batch, y_batch in loader:
        logits, _, _ = model(x_batch.to(device), apply_non_retention=False)
        probabilities.append(logits.softmax(dim=-1).cpu())
        targets.append(y_batch)
    probability = torch.cat(probabilities).numpy()
    target = torch.cat(targets).numpy()
    prediction = probability.argmax(axis=1)
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro")),
    }
    try:
        result["macro_auroc"] = float(
            roc_auc_score(
                target,
                probability,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        result["macro_auroc"] = None
    return result


def load_npz(path: str) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    arrays = np.load(path, allow_pickle=False)
    if "x_train" not in arrays or "y_train" not in arrays:
        raise ValueError("NPZ must contain x_train and y_train")
    x_train = torch.from_numpy(np.asarray(arrays["x_train"], dtype=np.float32))
    y_train = torch.from_numpy(np.asarray(arrays["y_train"], dtype=np.int64))
    x_test = (
        torch.from_numpy(np.asarray(arrays["x_test"], dtype=np.float32))
        if "x_test" in arrays
        else None
    )
    y_test = (
        torch.from_numpy(np.asarray(arrays["y_test"], dtype=np.int64))
        if "y_test" in arrays
        else None
    )
    return x_train, y_train, x_test, y_test


def save_run(
    output_dir: str,
    model: NSPrivacyNet,
    config: TrainConfig,
    audit: dict,
    evaluation: Optional[dict] = None,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination / "model_state.pt")
    record = {
        "config": asdict(config),
        "privacy_audit": audit,
        "evaluation": evaluation,
    }
    (destination / "run_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
