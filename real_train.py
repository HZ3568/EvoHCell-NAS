"""异构DARTS实验主入口

提供:
1. 完整的进化搜索管道 (NSGA-II)
2. 发现架构的训练
3. 实验管理和日志记录
4. 命令行接口

使用方法:
    # 运行完整实验 (搜索 + 训练)
    python real_train.py

    # 仅运行进化搜索
    python real_train.py --mode search --generations 20

    # 指定配置文件
    python real_train.py --config experiments/config.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset

from darts.model_hetero_cell import NetworkCIFARHeteroCell
from darts.utils import _data_transforms_cifar10, accuracy
from genetic.population import Individual


# ============================================================================
# 配置加载
# ============================================================================

CONFIG_PATH = Path(__file__).parent / "experiments" / "config.json"


def load_config(config_file: str = None) -> Dict[str, Any]:
    """加载配置文件

    Args:
        config_file: 配置文件路径，None则使用默认配置

    Returns:
        配置字典
    """
    path = Path(config_file) if config_file else CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_evolution_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """从原始配置构建进化搜索配置"""
    evo = raw.get("进化搜索", {}).copy()
    evo["layers"] = raw.get("架构", {}).get("layers", 8)
    return evo


def build_train_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """从原始配置构建训练配置"""
    cfg = {}
    cfg.update(raw.get("架构", {}))
    cfg.update(raw.get("训练", {}))
    cfg.update(raw.get("数据", {}))
    cfg["save_dir"] = raw.get("系统", {}).get("save_dir", "./results") + "/final_train"
    return cfg


# ============================================================================
# 训练结果
# ============================================================================

@dataclass
class TrainResult:
    best_valid_top1: float
    final_valid_top1: float
    best_epoch: int
    report_path: str
    checkpoint_path: str


# ============================================================================
# 架构选择
# ============================================================================

def select_representative_individual(individuals: List[Individual], fronts: List[List[int]]) -> Individual:
    """从Pareto前沿选择代表性个体"""
    if not individuals:
        raise ValueError("种群为空")
    if not fronts:
        return min(individuals, key=lambda x: float(sum(x.fitness)))
    first_front = fronts[0]
    if not first_front:
        return min(individuals, key=lambda x: float(sum(x.fitness)))
    return min((individuals[i] for i in first_front), key=lambda x: float(sum(x.fitness)))


# ============================================================================
# 数据加载
# ============================================================================

def _build_loaders(cfg: Dict[str, Any], device: torch.device):
    """构建CIFAR-10数据加载器"""
    train_tf, valid_tf = _data_transforms_cifar10(
        cutout=bool(cfg.get("cutout", True)),
        cutout_length=int(cfg.get("cutout_length", 16)),
    )
    data_root = str(cfg.get("data_root", "genetic/data"))
    train_data = dset.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    valid_data = dset.CIFAR10(root=data_root, train=False, download=True, transform=valid_tf)
    pin_memory = device.type == "cuda"
    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=int(cfg.get("batch_size", 96)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 2)),
        pin_memory=pin_memory,
    )
    valid_queue = torch.utils.data.DataLoader(
        valid_data,
        batch_size=int(cfg.get("eval_batch_size", 256)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
        pin_memory=pin_memory,
    )
    return train_queue, valid_queue


# ============================================================================
# 训练和验证
# ============================================================================

def _train_one_epoch(
        model: nn.Module,
        train_queue,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        drop_path_prob: float,
        max_steps: Optional[int],
        auxiliary_weight: float = 0.0,
) -> Tuple[float, float]:
    """训练一个epoch"""
    model.train()
    model.drop_path_prob = drop_path_prob
    loss_sum = 0.0
    top1_sum = 0.0
    n_sum = 0
    for step, (inputs, targets) in enumerate(train_queue):
        if max_steps is not None and step >= max_steps:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # 处理辅助头输出
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            logits, logits_aux = outputs
            loss = criterion(logits, targets)
            if auxiliary_weight > 0:
                loss_aux = criterion(logits_aux, targets)
                loss = loss + auxiliary_weight * loss_aux
        else:
            logits = outputs
            loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()
        prec1 = accuracy(logits, targets, topk=(1,))[0]
        bs = int(inputs.size(0))
        loss_sum += float(loss.item()) * bs
        top1_sum += float(prec1.item()) * bs
        n_sum += bs
    if n_sum == 0:
        return 0.0, 0.0
    return loss_sum / n_sum, top1_sum / n_sum


@torch.no_grad()
def _valid_one_epoch(
        model: nn.Module,
        valid_queue,
        criterion: nn.Module,
        device: torch.device,
        max_steps: Optional[int],
) -> Tuple[float, float]:
    """验证一个epoch"""
    model.eval()
    model.drop_path_prob = 0.0
    loss_sum = 0.0
    top1_sum = 0.0
    n_sum = 0
    for step, (inputs, targets) in enumerate(valid_queue):
        if max_steps is not None and step >= max_steps:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        prec1 = accuracy(logits, targets, topk=(1,))[0]
        bs = int(inputs.size(0))
        loss_sum += float(loss.item()) * bs
        top1_sum += float(prec1.item()) * bs
        n_sum += bs
    if n_sum == 0:
        return 0.0, 0.0
    return loss_sum / n_sum, top1_sum / n_sum


# ============================================================================
# 模型训练
# ============================================================================

def train_individual_with_cifar10(individual: Individual, cfg: Dict[str, Any]) -> TrainResult:
    """使用CIFAR-10训练单个个体"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = str(cfg.get("save_dir", os.path.join(os.path.dirname(__file__), "results", "final_train")))
    os.makedirs(save_dir, exist_ok=True)
    train_queue, valid_queue = _build_loaders(cfg, device)

    # 辅助头配置
    auxiliary = bool(cfg.get("auxiliary", False))
    auxiliary_weight = float(cfg.get("auxiliary_weight", 0.4)) if auxiliary else 0.0

    model = NetworkCIFARHeteroCell(
        C=int(cfg.get("init_channels", 16)),
        num_classes=10,
        layers=int(cfg.get("layers", 8)),
        auxiliary=auxiliary,
        genotype_list=individual.genotype,
    ).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 0.025)),
        momentum=float(cfg.get("momentum", 0.9)),
        weight_decay=float(cfg.get("weight_decay", 3e-4)),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.get("epochs", 50)),
        eta_min=float(cfg.get("min_learning_rate", 1e-3)),
    )
    epochs = int(cfg.get("epochs", 50))
    max_train_steps = cfg.get("max_train_steps_per_epoch")
    max_valid_steps = cfg.get("max_valid_steps_per_epoch")
    max_train_steps = int(max_train_steps) if max_train_steps is not None else None
    max_valid_steps = int(max_valid_steps) if max_valid_steps is not None else None
    max_drop_path_prob = float(cfg.get("drop_path_prob", 0.2))
    best_valid_top1 = -1.0
    final_valid_top1 = -1.0
    best_epoch = 0
    best_state = None
    logs: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        drop_path = max_drop_path_prob * float(epoch + 1) / float(epochs)
        train_loss, train_top1 = _train_one_epoch(
            model=model,
            train_queue=train_queue,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            drop_path_prob=drop_path,
            max_steps=max_train_steps,
            auxiliary_weight=auxiliary_weight,
        )
        valid_loss, valid_top1 = _valid_one_epoch(
            model=model,
            valid_queue=valid_queue,
            criterion=criterion,
            device=device,
            max_steps=max_valid_steps,
        )
        scheduler.step()
        final_valid_top1 = valid_top1
        if valid_top1 > best_valid_top1:
            best_valid_top1 = valid_top1
            best_epoch = epoch + 1
            best_state = {
                "model": model.state_dict(),
                "epoch": best_epoch,
                "best_valid_top1": best_valid_top1,
                "individual_fitness": list(individual.fitness),
                "train_config": cfg,
                "genotype_list": individual.genotype,
            }
        logs.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_top1": train_top1,
                "valid_loss": valid_loss,
                "valid_top1": valid_top1,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"[训练 {epoch + 1}/{epochs}] "
            f"训练损失={train_loss:.4f} 训练准确率={train_top1:.2f} "
            f"验证损失={valid_loss:.4f} 验证准确率={valid_top1:.2f}"
        )
    elapsed_sec = time.perf_counter() - t0
    ckpt_path = os.path.join(save_dir, "best_model.pth")
    if best_state is None:
        best_state = {
            "model": model.state_dict(),
            "epoch": 0,
            "best_valid_top1": final_valid_top1,
            "individual_fitness": list(individual.fitness),
            "train_config": cfg,
            "genotype_list": individual.genotype,
        }
    torch.save(best_state, ckpt_path)
    report = {
        "best_valid_top1": best_valid_top1,
        "final_valid_top1": final_valid_top1,
        "best_epoch": best_epoch,
        "elapsed_sec": elapsed_sec,
        "individual_fitness": list(individual.fitness),
        "logs": logs,
    }
    report_path = os.path.join(save_dir, "train_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return TrainResult(
        best_valid_top1=best_valid_top1,
        final_valid_top1=final_valid_top1,
        best_epoch=best_epoch,
        report_path=report_path,
        checkpoint_path=ckpt_path,
    )


# ============================================================================
# 主实验管道
# ============================================================================

def run_evolutionary_search(config: Dict[str, Any]) -> Tuple[Any, List[List[int]]]:
    """运行NSGA-II进化搜索"""
    from genetic.NSGA_II import run_nsga2

    print("\n" + "=" * 80)
    print("开始进化搜索 (NSGA-II)")
    print("=" * 80)
    print(f"配置:")
    print(f"  - 代数: {config.get('generations', 20)}")
    print(f"  - 种群大小: {config.get('population_size', 50)}")
    print(f"  - 层级交叉概率: {config.get('pc_layer', 0.5)}")
    print(f"  - 层级变异概率: {config.get('pm_layer', 0.15)}")
    print(f"  - 零样本指标: {config.get('metric', 'synflow')}")
    print(f"  - 随机种子: {config.get('seed', 42)}")
    print()

    t0 = time.perf_counter()
    final_pop, fronts = run_nsga2(config)
    elapsed = time.perf_counter() - t0

    print(f"\n进化搜索完成，耗时 {elapsed:.2f}s")
    print(f"最终种群大小: {len(final_pop.individuals)}")
    print(f"Pareto前沿数: {len(fronts)}")
    print(f"第一前沿大小: {len(fronts[0]) if fronts else 0}")

    return final_pop, fronts


def run_full_experiment(
        evolution_config: Dict[str, Any],
        train_config: Dict[str, Any],
        save_dir: str = "./results/experiment"
) -> Dict[str, Any]:
    """运行完整实验: 搜索 + 训练最佳架构"""
    os.makedirs(save_dir, exist_ok=True)

    # 保存配置
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "evolution": evolution_config,
            "train": train_config,
        }, f, indent=2)

    # 第1步: 进化搜索
    final_pop, fronts = run_evolutionary_search(evolution_config)

    # 保存搜索结果
    search_results_dir = os.path.join(save_dir, "search_results")
    os.makedirs(search_results_dir, exist_ok=True)

    # 保存Pareto前沿
    pareto_path = os.path.join(search_results_dir, "pareto_front.json")
    first_front_data = []
    if fronts:
        for idx in fronts[0]:
            ind = final_pop.individuals[idx]
            first_front_data.append({
                "fitness": list(ind.fitness),
                "genotype_list": [str(g) for g in ind.genotype],
            })

    with open(pareto_path, "w", encoding="utf-8") as f:
        json.dump(first_front_data, f, indent=2)

    print(f"\nPareto前沿已保存到: {pareto_path}")

    # 第2步: 选择最佳个体
    best_individual = select_representative_individual(final_pop.individuals, fronts)
    print(f"\n选中最佳个体:")
    print(f"  - 适应度: {best_individual.fitness}")

    # 保存最佳基因型
    best_genotype_path = os.path.join(search_results_dir, "best_genotype.txt")
    with open(best_genotype_path, "w", encoding="utf-8") as f:
        for i, genotype in enumerate(best_individual.genotype):
            f.write(f"# 第{i}层\n")
            f.write(str(genotype) + "\n")

    print(f"最佳基因型已保存到: {best_genotype_path}")

    # 第3步: 训练最佳架构
    print("\n" + "=" * 80)
    print("训练最佳架构")
    print("=" * 80)

    train_config_copy = train_config.copy()
    train_config_copy["save_dir"] = os.path.join(save_dir, "training")

    train_result = train_individual_with_cifar10(best_individual, train_config_copy)

    print(f"\n训练完成!")
    print(f"  - 最佳验证准确率: {train_result.best_valid_top1:.2f}%")
    print(f"  - 最终验证准确率: {train_result.final_valid_top1:.2f}%")
    print(f"  - 最佳epoch: {train_result.best_epoch}")

    # 创建摘要
    summary = {
        "搜索": {
            "最终种群大小": len(final_pop.individuals),
            "前沿数": len(fronts),
            "第一前沿大小": len(fronts[0]) if fronts else 0,
            "最佳适应度": list(best_individual.fitness),
        },
        "训练": {
            "最佳验证准确率": train_result.best_valid_top1,
            "最终验证准确率": train_result.final_valid_top1,
            "最佳epoch": train_result.best_epoch,
        },
        "路径": {
            "配置": config_path,
            "pareto前沿": pareto_path,
            "最佳基因型": best_genotype_path,
            "检查点": train_result.checkpoint_path,
            "训练报告": train_result.report_path,
        }
    }

    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n实验摘要已保存到: {summary_path}")
    print("\n" + "=" * 80)
    print("实验完成!")
    print("=" * 80)

    return summary


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="异构DARTS进化搜索实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python real_train.py                              # 运行完整实验
  python real_train.py --generations 30 --seed 123  # 自定义参数
  python real_train.py --mode search                # 仅搜索
  python real_train.py --config experiments/config.json
        """
    )

    parser.add_argument("--mode", type=str, default="full", choices=["full", "search", "train"],
                        help="实验模式: full(搜索+训练), search(仅搜索), train(仅训练)")
    parser.add_argument("--generations", type=int, default=None, help="进化代数")
    parser.add_argument("--population_size", type=int, default=None, help="种群大小")
    parser.add_argument("--pc_layer", type=float, default=None, help="层级交叉概率")
    parser.add_argument("--pm_layer", type=float, default=None, help="层级变异概率")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--metric", type=str, default=None, choices=["synflow", "grad_norm"],
                        help="零样本评估指标")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--save_dir", type=str, default=None, help="结果保存目录")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")

    args = parser.parse_args()

    # 加载配置
    raw_config = load_config(args.config)
    evolution_config = build_evolution_config(raw_config)
    train_config = build_train_config(raw_config)

    # 命令行参数覆盖
    if args.generations is not None:
        evolution_config["generations"] = args.generations
    if args.population_size is not None:
        evolution_config["population_size"] = args.population_size
    if args.pc_layer is not None:
        evolution_config["pc_layer"] = args.pc_layer
    if args.pm_layer is not None:
        evolution_config["pm_layer"] = args.pm_layer
    if args.seed is not None:
        evolution_config["seed"] = args.seed
    if args.metric is not None:
        evolution_config["metric"] = args.metric
    if args.epochs is not None:
        train_config["epochs"] = args.epochs

    # 确定保存目录
    if args.save_dir:
        save_dir = args.save_dir
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = f"./results/exp_{timestamp}"

    # 根据模式运行
    if args.mode == "full":
        run_full_experiment(evolution_config, train_config, save_dir)
    elif args.mode == "search":
        final_pop, fronts = run_evolutionary_search(evolution_config)
        os.makedirs(save_dir, exist_ok=True)
        pareto_path = os.path.join(save_dir, "pareto_front.json")
        first_front_data = []
        if fronts:
            for idx in fronts[0]:
                ind = final_pop.individuals[idx]
                first_front_data.append({
                    "fitness": list(ind.fitness),
                    "genotype_list": [str(g) for g in ind.genotype],
                })
        with open(pareto_path, "w", encoding="utf-8") as f:
            json.dump(first_front_data, f, indent=2)
        print(f"\n搜索结果已保存到: {save_dir}")
    elif args.mode == "train":
        print("仅训练模式尚未实现，请使用 'full' 或 'search' 模式")
        sys.exit(1)


if __name__ == "__main__":
    main()
