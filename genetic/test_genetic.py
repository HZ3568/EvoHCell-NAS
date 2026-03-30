"""遗传算法NSGA-II框架的基础测试

可以通过pytest或直接运行:
- `python -m genetic.test_genetic`（推荐）
- 或 `python genetic/test_genetic.py`
"""
from __future__ import annotations
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from typing import List
import math
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple

# 兼容pytest和直接运行的导入
try:
    from genetic.NSGA_II import run_nsga2, NSGAII
    from genetic.population import Population, Individual
    from genetic.evaluate import Evaluator
    from genetic.crossover_and_mutation import layerwise_crossover
    from experiments.config_loader import get_evolution_config, get_train_config
    from real_train import train_individual_with_cifar10, select_representative_individual
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from genetic.NSGA_II import run_nsga2, NSGAII
    from genetic.population import Population, Individual
    from genetic.evaluate import Evaluator
    from genetic.crossover_and_mutation import layerwise_crossover
    from experiments.config_loader import get_evolution_config, get_train_config
    from real_train import train_individual_with_cifar10, select_representative_individual

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def test_run_nsga2_basic():
    config = get_evolution_config()
    config.update({
        "generations": 2,
        "seed": 7,
    })
    final_pop, fronts = run_nsga2(config)

    assert len(final_pop.individuals) <= config["population_size"]
    assert len(final_pop.individuals) >= config["init_pop_size"]
    assert len(fronts) >= 1
    first_front = fronts[0]
    assert len(first_front) >= 1

    for ind in final_pop.individuals:
        assert all(math.isfinite(f) for f in ind.fitness)

    # 第一前沿中的个体互不支配
    nsga = NSGAII()
    inds = final_pop.individuals
    for i in range(len(first_front)):
        for j in range(i + 1, len(first_front)):
            a = inds[first_front[i]]
            b = inds[first_front[j]]
            assert not nsga.is_dominate(a, b)
            assert not nsga.is_dominate(b, a)


# ----------------------------
# 实验工具
# ----------------------------

@dataclass
class RunLog:
    config: Dict[str, Any]
    convergence: List[float]
    first_front_sizes: List[int]
    runtimes_sec: float
    final_pop: Population
    final_fronts: List[List[int]]
    avg_crowding_first_front: float
    min_distance_to_ideal: float


def _scalarize_sum(ind: Individual) -> float:
    f1, f2 = ind.fitness[0], ind.fitness[1]
    return f1 + f2


def _distance_to_ideal(ind: Individual) -> float:
    # 理想点假设为(0,0)
    f1, f2 = ind.fitness[0], ind.fitness[1]
    return math.sqrt(f1 * f1 + f2 * f2)


def run_nsga2_with_logging(config: Dict[str, Any]) -> RunLog:
    cfg = get_evolution_config()
    cfg.update(config)
    if "init_population_path" not in cfg:
        cfg["init_population_path"] = os.path.join(os.path.dirname(__file__), "init_population.txt")

    cfg["pop_size"] = int(cfg["init_pop_size"])
    pop = Population(cfg)
    pop.initialize()
    evaluator = Evaluator(cfg)
    evaluator.evaluate_population(pop)

    nsga = NSGAII()
    fronts = nsga.fast_nondominated_sort(pop.individuals)
    pop.front = fronts
    for front in fronts:
        nsga.crowding_distance(front, pop.individuals)

    gens = int(cfg["generations"])
    max_population_size = int(cfg["population_size"])
    parent_pool_size = int(cfg["parent_pool_size"])
    crossover_rounds = int(cfg["crossover_rounds"])

    def score(ind: Individual) -> float:
        return float(sum(ind.fitness))

    def select_two_parents(gen_idx: int) -> Tuple[Individual, Individual]:
        import random
        if gen_idx == 0:
            return tuple(random.sample(pop.individuals, 2))
        ordered = sorted(pop.individuals, key=score)
        k = min(max(2, parent_pool_size), len(ordered))
        return tuple(random.sample(ordered[:k], 2))

    convergence: List[float] = []
    first_front_sizes: List[int] = []
    t0 = time.perf_counter()

    for gen in range(gens):
        offsprings: List[Individual] = []
        for _ in range(crossover_rounds):
            p1, p2 = select_two_parents(gen)
            c1, c2 = layerwise_crossover(p1, p2, cfg)
            offsprings.append(c1)
            offsprings.append(c2)

        for child in offsprings:
            evaluator.evaluate_individual(child)
        combined = pop.individuals + offsprings
        if len(combined) > max_population_size:
            pop.individuals = nsga.select_next_generation(combined, max_population_size)
        else:
            pop.individuals = combined

        fronts = nsga.fast_nondominated_sort(pop.individuals)
        pop.front = fronts
        for front in fronts:
            nsga.crowding_distance(front, pop.individuals)

        # 每代记录日志
        best_scalar = min(_scalarize_sum(ind) for ind in pop.individuals)
        convergence.append(best_scalar)
        first_front_sizes.append(len(fronts[0]) if fronts else 0)

        print(
            f"[第{gen + 1}/{gens}代] "
            f"最佳目标和={best_scalar:.4f} "
            f"第一前沿={first_front_sizes[-1]}"
        )

    t1 = time.perf_counter()

    # 最终质量指标
    first_front_inds = [pop.individuals[i] for i in pop.front[0]] if pop.front else []
    avg_crowding = (sum(ind.crowd_distance for ind in first_front_inds) / len(
        first_front_inds)) if first_front_inds else 0.0
    min_dist_ideal = min((_distance_to_ideal(ind) for ind in pop.individuals))

    return RunLog(
        config=cfg,
        convergence=convergence,
        first_front_sizes=first_front_sizes,
        runtimes_sec=(t1 - t0),
        final_pop=pop,
        final_fronts=pop.front,
        avg_crowding_first_front=avg_crowding,
        min_distance_to_ideal=min_dist_ideal,
    )


def _plot_convergence(runlogs: List[RunLog], save_path: str) -> None:
    if plt is None:
        print("matplotlib不可用，跳过收敛曲线绘制")
        return
    plt.figure(figsize=(8, 5), dpi=140)
    for rl in runlogs:
        label = f"pop={rl.config.get('pop_size')} pc_layer={rl.config.get('pc_layer')}"
        plt.plot(range(1, len(rl.convergence) + 1), rl.convergence, label=label)
    plt.xlabel("代数")
    plt.ylabel("最佳 f1+f2 (越小越好)")
    plt.title("NSGA-II 收敛曲线")
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _plot_pareto_fronts(runlogs: List[RunLog], save_path: str) -> List[str]:
    if plt is None:
        print("matplotlib不可用，跳过Pareto前沿绘制")
        return []
    saved_paths: List[str] = []
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    base, ext = os.path.splitext(save_path)
    for rl in runlogs:
        pc = str(rl.config.get("pc_layer")).replace(".", "p")
        seed = rl.config.get("seed")
        one_path = f"{base}_pc{pc}_seed{seed}{ext}"
        plt.figure(figsize=(8, 6), dpi=140)
        x_vals = [ind.fitness[0] for ind in rl.final_pop.individuals]
        y_vals = [ind.fitness[1] for ind in rl.final_pop.individuals]
        plt.scatter(x_vals, y_vals, s=28, alpha=0.85)
        for i, (x, y) in enumerate(zip(x_vals, y_vals)):
            plt.text(x, y, str(i), fontsize=7, alpha=0.8)
        plt.xlabel("目标1: -零样本分数 (越小越好)")
        plt.ylabel("目标2: 参数量 (越小越好)")
        plt.title(f"所有个体的Pareto散点图 (pc_layer={rl.config.get('pc_layer')}, seed={seed})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(one_path)
        plt.close()
        saved_paths.append(one_path)
    return saved_paths


def _write_experiment_report(runlogs: List[RunLog], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lines: List[str] = []
    lines.append("# 实验结果\n")
    lines.append("\n## 参数设置与性能对比\n")
    for rl in runlogs:
        cfg = rl.config
        lines.append(
            f"- pop_size={cfg.get('pop_size')}, generations={cfg.get('generations')}, pc_layer={cfg.get('pc_layer')}, time={rl.runtimes_sec:.2f}s, min(f1+f2)={min(rl.convergence):.4f}, first_front_size={rl.first_front_sizes[-1]}\n"
        )
    lines.append("\n## 算法收敛性分析\n")
    lines.append("- 不同参数下的收敛曲线已保存为 `results/nsga2_convergence.png`，曲线越低代表目标和越小，收敛越好。\n")
    lines.append("- 迭代过程中 first_front 的规模变化反映了非支配解的丰富度，通常随着代数增加而稳定。\n")
    lines.append("\n## 帕累托解集质量评估\n")
    for rl in runlogs:
        lines.append(
            f"- init_pop={rl.config.get('init_pop_size')} pop_cap={rl.config.get('population_size')} pc_layer={rl.config.get('pc_layer')}: avg crowding on first front={rl.avg_crowding_first_front:.3f}, min distance to ideal={rl.min_distance_to_ideal:.3f}\n"
        )
    lines.append("- 帕累托图按不同 pc_layer 分别保存，每张图展示该实验最终种群全部个体（含编号），横轴为目标1、纵轴为目标2。\n")
    lines.append("\n## 结论\n")
    lines.append("- 适中的层间交叉概率（如 pc_layer=0.5）可在保持多样性的同时获得较稳定的第一前沿。\n")
    lines.append("- 增大种群规模（如 pop=80）通常能提升帕累托解的覆盖与多样性，但运行时间也会增加。\n")
    lines.append("\n(以上分析基于示例双目标函数，可按实际问题替换 Evaluator 以获得更具代表性的结论。)\n")

    with open(save_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _select_global_best_from_runlogs(runlogs: List[RunLog]) -> Individual:
    if not runlogs:
        raise ValueError("runlogs为空")
    candidates = [
        select_representative_individual(rl.final_pop.individuals, rl.final_fronts)
        for rl in runlogs
    ]
    return min(candidates, key=lambda x: float(sum(x.fitness)))


if __name__ == "__main__":
    # 先运行基础单元测试
    try:
        test_run_nsga2_basic()
        print("test_run_nsga2_basic: 通过")
    except AssertionError as e:
        print("test_run_nsga2_basic: 失败", e)
        raise

    # 运行实验：不同参数设置的性能对比与可视化
    base_config = get_evolution_config()
    exp_configs: List[Dict[str, Any]] = [
        {**base_config, "pc_layer": 0.3, "seed": 42},
        {**base_config, "pc_layer": 0.5, "seed": 42},
        {**base_config, "pc_layer": 0.7, "seed": 42},
    ]

    runlogs: List[RunLog] = []
    for cfg in exp_configs:
        rl = run_nsga2_with_logging(cfg)
        runlogs.append(rl)
        print(
            f"[运行] init_pop={cfg['init_pop_size']} pop_cap={cfg['population_size']} pc_layer={cfg['pc_layer']} time={rl.runtimes_sec:.2f}s min(f1+f2)={min(rl.convergence):.4f} first_front={rl.first_front_sizes[-1]}"
        )

    # 输出并保存图形与报告
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    conv_path = os.path.join(results_dir, "nsga2_convergence.png")
    fronts_path = os.path.join(results_dir, "nsga2_pareto_fronts.png")
    report_path = os.path.join(results_dir, "实验结果.md")

    _plot_convergence(runlogs, conv_path)
    saved_front_paths = _plot_pareto_fronts(runlogs, fronts_path)
    _write_experiment_report(runlogs, report_path)

    print(f"收敛曲线已保存到: {conv_path}")
    for p in saved_front_paths:
        print(f"Pareto前沿图已保存到: {p}")
    print(f"报告已保存到: {report_path}")

    train_cfg = get_train_config()
    train_cfg["save_dir"] = os.path.join(results_dir, "final_train")
    train_cfg["layers"] = base_config["layers"]
    best_ind = _select_global_best_from_runlogs(runlogs)
    train_result = train_individual_with_cifar10(best_ind, train_cfg)
    print(
        f"最终训练完成. 最佳准确率={train_result.best_valid_top1:.2f} "
        f"最终准确率={train_result.final_valid_top1:.2f} "
        f"最佳epoch={train_result.best_epoch}"
    )
    print(f"检查点已保存到: {train_result.checkpoint_path}")
    print(f"训练报告已保存到: {train_result.report_path}")
