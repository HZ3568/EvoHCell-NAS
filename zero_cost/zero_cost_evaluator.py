import traceback
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset

import zero_cost.grad_norm
import zero_cost.synflow
from darts.model import NetworkCIFAR
from darts.genotypes import Genotype
from darts.utils import _data_transforms_cifar10
from zero_cost.zero_utils import _measure_impls


# =========================
# Configuration
# =========================

@dataclass
class EvalConfig:
    metric: str = "synflow"          # grad_norm / synflow / synflow_bn ...
    batch_size: int = 64
    data_root: str = "./data"
    C: int = 16
    num_classes: int = 10
    layers: int = 8
    auxiliary: bool = True
    use_cuda_benchmark: bool = True


@dataclass
class MetricSpec:
    requires_data: bool
    model_mode: str                  # "train" or "eval"
    needs_loss_fn: bool


@dataclass
class ZeroCostResult:
    metric: str
    total_score: float
    non_zero_layers: int
    total_layers: int
    layer_scores: List[float]
    device: str
    input_shape: List[int]
    success: bool
    message: str = ""


METRIC_SPECS: Dict[str, MetricSpec] = {
    "grad_norm": MetricSpec(
        requires_data=True,
        model_mode="train",
        needs_loss_fn=True
    ),
    "synflow": MetricSpec(
        requires_data=False,
        model_mode="eval",
        needs_loss_fn=False
    ),
    "synflow_bn": MetricSpec(
        requires_data=False,
        model_mode="eval",
        needs_loss_fn=False
    ),
}


# =========================
# Genotype Factory
# =========================

def build_demo_genotype() -> Genotype:
    """
    Return a demo genotype for evaluation.
    Replace this with your searched subnet genotype when needed.
    """
    return Genotype(
        normal=[
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('skip_connect', 0), ('dil_conv_3x3', 2)
        ],
        normal_concat=[2, 3, 4, 5],
        reduce=[
            ('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('skip_connect', 2), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 2), ('max_pool_3x3', 1)
        ],
        reduce_concat=[2, 3, 4, 5]
    )


# =========================
# Utility Functions
# =========================

def validate_metric(metric: str) -> None:
    if metric not in _measure_impls:
        raise ValueError(
            f"Unsupported metric: {metric}. "
            f"Available metrics: {list(_measure_impls.keys())}"
        )
    if metric not in METRIC_SPECS:
        raise ValueError(
            f"Metric '{metric}' is not configured in METRIC_SPECS. "
            f"Please add its behavior definition first."
        )


def build_device(use_cuda_benchmark: bool = True) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = use_cuda_benchmark
    else:
        device = torch.device("cpu")
    return device


def build_model(genotype: Genotype, config: EvalConfig, device: torch.device) -> nn.Module:
    model = NetworkCIFAR(
        C=config.C,
        num_classes=config.num_classes,
        layers=config.layers,
        auxiliary=config.auxiliary,
        genotype=genotype
    )
    return model.to(device)


def get_cifar10_batch(batch_size: int = 64, data_root: str = "./data"):
    """
    Load one batch from CIFAR-10.
    Priority: try download=True, fallback to local existing data.
    """
    train_transform, _ = _data_transforms_cifar10(cutout=False, cutout_length=0)

    try:
        train_data = dset.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=train_transform
        )
    except Exception as e:
        print(f"[WARN] Dataset download failed: {e}")
        print("[INFO] Trying to use existing local CIFAR-10 data...")
        train_data = dset.CIFAR10(
            root=data_root,
            train=True,
            download=False,
            transform=train_transform
        )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )

    inputs, targets = next(iter(train_loader))
    return inputs, targets


def prepare_inputs(metric: str, config: EvalConfig, device: torch.device):
    """
    Prepare metric-specific input/target tensors.
    """
    spec = METRIC_SPECS[metric]

    if spec.requires_data:
        inputs, targets = get_cifar10_batch(
            batch_size=config.batch_size,
            data_root=config.data_root
        )
        inputs = inputs.to(device)
        targets = targets.to(device)
    else:
        inputs = torch.ones(1, 3, 32, 32, device=device)
        targets = torch.zeros(1, dtype=torch.long, device=device)

    return inputs, targets


def set_model_mode(model: nn.Module, metric: str) -> None:
    spec = METRIC_SPECS[metric]
    if spec.model_mode == "eval":
        model.eval()
    elif spec.model_mode == "train":
        model.train()
    else:
        raise ValueError(f"Unsupported model mode: {spec.model_mode}")


def compute_zero_cost_score(
    model: nn.Module,
    device: torch.device,
    metric: str,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Optional[nn.Module] = None
) -> ZeroCostResult:
    """
    Run one zero-cost metric and aggregate layer-wise scores.
    """
    validate_metric(metric)

    spec = METRIC_SPECS[metric]

    if spec.needs_loss_fn and loss_fn is None:
        raise ValueError(f"Metric '{metric}' requires a loss function, but loss_fn is None.")

    if spec.needs_loss_fn:
        raw_scores = _measure_impls[metric](model, device, inputs, targets, loss_fn)
    else:
        raw_scores = _measure_impls[metric](model, device, inputs, targets)

    layer_scores = [score.sum().item() for score in raw_scores]
    non_zero_layers = sum(1 for value in layer_scores if value > 0)
    total_score = sum(layer_scores)

    return ZeroCostResult(
        metric=metric,
        total_score=total_score,
        non_zero_layers=non_zero_layers,
        total_layers=len(layer_scores),
        layer_scores=layer_scores,
        device=str(device),
        input_shape=list(inputs.shape),
        success=True,
        message="Evaluation successful."
    )


def evaluate_genotype(genotype: Genotype, config: EvalConfig) -> ZeroCostResult:
    """
    End-to-end evaluation pipeline:
    1. build device
    2. build model
    3. prepare input
    4. set model mode
    5. run metric
    """
    validate_metric(config.metric)

    device = build_device(config.use_cuda_benchmark)
    model = build_model(genotype, config, device)
    inputs, targets = prepare_inputs(config.metric, config, device)
    set_model_mode(model, config.metric)

    loss_fn = nn.CrossEntropyLoss()

    return compute_zero_cost_score(
        model=model,
        device=device,
        metric=config.metric,
        inputs=inputs,
        targets=targets,
        loss_fn=loss_fn
    )


def print_result(result: ZeroCostResult) -> None:
    print("-" * 50)
    print("Zero-Cost Evaluation Result")
    print("-" * 50)
    print(f"Metric                : {result.metric}")
    print(f"Device                : {result.device}")
    print(f"Input Shape           : {result.input_shape}")
    print(f"Layers Evaluated      : {result.total_layers}")
    print(f"Non-zero Layers       : {result.non_zero_layers}")
    print(f"Total Score           : {result.total_score:.6f}")
    print(f"Success               : {result.success}")
    print(f"Message               : {result.message}")

    if result.total_score == 0.0:
        print("\n[WARNING] Score is 0.0. Possible reasons:")
        print("1. Backward pass did not populate gradients or scores.")
        print("2. Metric setting is incompatible with the model state.")
        print("3. Try switching between synflow and synflow_bn.")
        print("4. Check whether the metric implementation matches this network.")
    else:
        print("\nEvaluation completed successfully.")

    print("-" * 50)


# =========================
# Optional: Batch Evaluation
# =========================

def evaluate_multiple_genotypes(
    genotype_dict: Dict[str, Genotype],
    config: EvalConfig
) -> Dict[str, ZeroCostResult]:
    """
    Evaluate multiple candidate subnets.
    Useful for NAS / subnet ranking experiments.
    """
    results = {}

    for name, genotype in genotype_dict.items():
        try:
            print(f"\n[INFO] Evaluating genotype: {name}")
            result = evaluate_genotype(genotype, config)
            results[name] = result
        except Exception as e:
            results[name] = ZeroCostResult(
                metric=config.metric,
                total_score=0.0,
                non_zero_layers=0,
                total_layers=0,
                layer_scores=[],
                device="unknown",
                input_shape=[],
                success=False,
                message=f"Evaluation failed: {e}"
            )
            print(f"[ERROR] Evaluation failed for {name}: {e}")
            traceback.print_exc()

    return results


# =========================
# Main
# =========================

def main():
    config = EvalConfig(
        metric="synflow",      # change to "grad_norm" if needed
        batch_size=64,
        data_root="./data",
        C=16,
        num_classes=10,
        layers=8,
        auxiliary=True
    )

    genotype = build_demo_genotype()

    print(f"Genotype: {genotype}")
    print(f"Metric: {config.metric}")

    try:
        result = evaluate_genotype(genotype, config)
        print_result(result)

        # If you need to convert to dict for saving/logging:
        result_dict = asdict(result)
        # print(result_dict)

    except Exception as e:
        print(f"[ERROR] Error during evaluation: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()