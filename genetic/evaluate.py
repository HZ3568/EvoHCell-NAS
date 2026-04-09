"""Fitness evaluation for genotype_list individuals."""

from __future__ import annotations
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torchvision.datasets as dset
import zero_cost.grad_norm
import zero_cost.synflow

from .population import Individual, Population
from darts.model_hetero_cell import NetworkCIFARHeteroCell
from darts.utils import data_transforms_cifar10
from zero_cost.zero_utils import _measure_impls


class Evaluator:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.metric = self.config.get("metric", "grad_norm")
        self.C = int(self.config.get("init_channels", 36))
        self.num_classes = int(self.config.get("num_classes", 10))
        self.layers = int(self.config.get("layers", 20))
        self.auxiliary = bool(self.config.get("auxiliary", True))
        self.maximize_score = bool(self.config.get("maximize_score", True))
        self.batch_size = int(self.config.get("batch_size", 64))
        self.data_root = self.config.get("data_root", "./data")
        device_cfg = self.config.get("device")
        if device_cfg is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_cfg)
        self.loss_fn = nn.CrossEntropyLoss()
        self._cached_inputs_targets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if self.metric not in _measure_impls:
            raise ValueError(f"Unsupported metric: {self.metric}. Available: {list(_measure_impls.keys())}")

    def _get_cifar10_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cached_inputs_targets is not None:
            return self._cached_inputs_targets
        train_transform, _ = data_transforms_cifar10(cutout=False, cutout_length=0)
        try:
            train_data = dset.CIFAR10(root=self.data_root, train=True, download=True, transform=train_transform)
        except Exception:
            train_data = dset.CIFAR10(root=self.data_root, train=True, download=False, transform=train_transform)
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=0
        )
        inputs, targets = next(iter(train_queue))
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        self._cached_inputs_targets = (inputs, targets)
        return self._cached_inputs_targets

    def _get_metric_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.metric == "grad_norm":
            return self._get_cifar10_batch()
        inputs = torch.ones(1, 3, 32, 32, device=self.device)
        targets = torch.zeros(1, dtype=torch.long, device=self.device)
        return inputs, targets

    def _compute_zero_cost_score(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        if self.metric == "grad_norm":
            raw_scores = _measure_impls[self.metric](model, self.device, inputs, targets, self.loss_fn)
        else:
            raw_scores = _measure_impls[self.metric](model, self.device, inputs, targets)
        layer_scores = [s.sum().item() for s in raw_scores]
        return float(sum(layer_scores))

    def evaluate_individual(self, indi: Individual) -> None:
        import math

        genotype_list = indi.genotype
        model = NetworkCIFARHeteroCell(
            C=self.C,
            num_classes=self.num_classes,
            layers=self.layers,
            auxiliary=self.auxiliary,
            genotype_list=genotype_list
        ).to(self.device)

        if self.metric in ("synflow", "synflow_bn"):
            model.eval()
        else:
            model.train()

        inputs, targets = self._get_metric_inputs()
        score = self._compute_zero_cost_score(model, inputs, targets)
        params = float(sum(p.numel() for p in model.parameters()))
        params_m = params / 1e6

        # 稳定化 zero-cost score
        if not math.isfinite(score):
            score_stable = -1e6
        elif self.metric in ("synflow", "synflow_bn"):
            if score <= 0:
                score_stable = -1e6
            else:
                score_stable = math.log1p(score)
        else:
            score_stable = math.copysign(math.log1p(abs(score)), score)

        obj1 = -score_stable if self.maximize_score else score_stable
        indi.fitness = [obj1, params_m]

    def evaluate_population(self, pop: Population) -> None:
        for indi in pop.individuals:
            self.evaluate_individual(indi)
