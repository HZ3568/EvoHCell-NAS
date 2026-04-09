from __future__ import annotations

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from darts.utils import create_exp_dir
from utils.logger_utils import setup_logger


def _available_metrics() -> list[str]:
    """获取可用的零代价指标名称。"""
    try:
        from zero_cost.zero_cost_evaluator import METRIC_SPECS
        return list(METRIC_SPECS.keys())
    except ImportError:
        return ["synflow", "grad_norm", "synflow_bn"]


def _coerce_args(args: argparse.Namespace | dict[str, Any]) -> argparse.Namespace:
    """将 dict 或 Namespace 统一转换为 Namespace。"""
    if isinstance(args, argparse.Namespace):
        return args
    if isinstance(args, dict):
        return argparse.Namespace(**args)
    raise TypeError("args 必须是 argparse.Namespace 或 dict")


def _normalize_args(args: argparse.Namespace | dict[str, Any]) -> argparse.Namespace:
    """
    将传入参数与 parser 默认参数进行合并。
    这样既支持命令行调用，也支持在 pipeline.py 中直接传 dict 调用。
    """
    parser = build_parser()
    defaults = parser.parse_args([])
    provided = _coerce_args(args)
    merged = vars(defaults).copy()
    merged.update(vars(provided))
    return argparse.Namespace(**merged)


def _validate_args(args: argparse.Namespace) -> None:
    """参数合法性检查。"""
    if args.generations <= 0:
        raise ValueError("generations 必须 > 0")
    if args.population_size <= 0:
        raise ValueError("population_size 必须 > 0")
    if args.top_k <= 0:
        raise ValueError("top_k 必须 > 0")
    if not (0.0 <= args.pc_layer <= 1.0):
        raise ValueError("pc_layer 必须在 [0, 1] 范围内")
    if not (0.0 <= args.pm_layer <= 1.0):
        raise ValueError("pm_layer 必须在 [0, 1] 范围内")
    if args.layers <= 0:
        raise ValueError("layers 必须 > 0")


def _set_random_seed(seed: int) -> None:
    """固定随机种子，尽量保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _edge_to_json(edge: Any) -> list[Any]:
    """将 genotype 中的一条边转换为 JSON 可序列化格式。"""
    if isinstance(edge, (list, tuple)) and len(edge) == 2:
        return [edge[0], int(edge[1])]
    raise ValueError(f"非法 edge 结构: {edge}")


def _genotype_to_json_dict(genotype: Any) -> dict[str, Any]:
    """
    将 Genotype 对象转换为字典，便于保存到 JSON。
    """
    required_attrs = ["normal", "normal_concat", "reduce", "reduce_concat"]
    if not all(hasattr(genotype, attr) for attr in required_attrs):
        raise TypeError(f"不支持的 genotype 类型: {type(genotype)}")

    return {
        "normal": [_edge_to_json(edge) for edge in genotype.normal],
        "normal_concat": [int(i) for i in genotype.normal_concat],
        "reduce": [_edge_to_json(edge) for edge in genotype.reduce],
        "reduce_concat": [int(i) for i in genotype.reduce_concat],
    }


def _edges_to_inline_str(edges: list[list[Any]]) -> str:
    """将边列表压缩成一行字符串，便于人工阅读。"""
    return "[" + ", ".join(f"({op},{idx})" for op, idx in edges) + "]"


def _genotype_to_readable_str(genotype_dict: dict[str, Any]) -> str:
    """将 genotype 转换为可读的字符串形式。"""
    normal_str = _edges_to_inline_str(genotype_dict["normal"])
    reduce_str = _edges_to_inline_str(genotype_dict["reduce"])
    return f"normal: {normal_str}\nreduce: {reduce_str}"


def _restore_zero_cost_score(fitness: list[float], maximize_score: bool) -> float:
    """
    恢复原始 zero-cost score。

    说明：
    - 若算法内部把“最大化 score”转成了“最小化 -score”，这里需要再转回来
    - fitness[0] 默认对应 zero-cost 目标
    """
    obj1 = float(fitness[0])
    return -obj1 if maximize_score else obj1


def _rank_candidate_indices(fronts: list[list[int]], total_count: int) -> list[int]:
    """
    根据帕累托分层结果生成候选排序。

    逻辑说明：
    - NSGA-II 会将种群分成若干层 Pareto fronts
    - front 0 表示第一非支配前沿，质量最好
    - front 1 表示第二前沿，以此类推
    - 这里按照 front 的先后顺序拼接成最终候选顺序
    """
    if not fronts:
        return list(range(total_count))

    ranked: list[int] = []
    for front in fronts:
        ranked.extend(front)

    seen = set(ranked)
    for idx in range(total_count):
        if idx not in seen:
            ranked.append(idx)

    return ranked


def _build_search_config(args: argparse.Namespace) -> dict[str, Any]:
    """
    将 argparse 参数整理成 NSGA-II 搜索所需配置。
    """
    return {
        "data": args.data,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "report_freq": args.report_freq,
        "generations": args.generations,
        "population_size": args.population_size,
        "pc_layer": args.pc_layer,
        "pm_layer": args.pm_layer,
        "seed": args.seed,
        "metric": args.metric,
        "layers": args.layers,
        "init_channels": args.init_channels,
        "maximize_score": args.maximize_score,
    }


def _plot_pareto_front(
        final_pop: Any,
        fronts: list[list[int]],
        save_dir: str,
        maximize_score: bool,
) -> None:
    """
    绘制帕累托前沿图并保存。

    横轴：参数量（MB），越小越好
    纵轴：zero-cost objective（NSGA-II 内部最小化目标），越小越好

    图中说明：
    - 灰色点：最终种群中的所有个体
    - 红色点：第一帕累托前沿（最优非支配解集合）

    注意：此图严格按照"两个目标都最小化"的视角绘制，
    因此前沿应朝左下方向延伸。
    """
    all_x = []
    all_y = []
    for ind in final_pop.individuals:
        all_x.append(float(ind.fitness[1]))  # params_mb
        all_y.append(float(ind.fitness[0]))  # zero-cost objective (内部最小化目标)

    plt.figure(figsize=(8, 6))
    plt.scatter(all_x, all_y, alpha=0.6, label="All Individuals")

    if fronts and len(fronts[0]) > 0:
        first_front = fronts[0]
        front_x = [float(final_pop.individuals[idx].fitness[1]) for idx in first_front]
        front_y = [float(final_pop.individuals[idx].fitness[0]) for idx in first_front]

        # 为了让前沿线更清晰，按参数量从小到大排序
        sorted_pairs = sorted(zip(front_x, front_y), key=lambda x: x[0])
        front_x_sorted = [p[0] for p in sorted_pairs]
        front_y_sorted = [p[1] for p in sorted_pairs]

        plt.scatter(front_x_sorted, front_y_sorted, s=50, label="Pareto Front")
        plt.plot(front_x_sorted, front_y_sorted)

    plt.xlabel("Params (MB)")
    plt.ylabel("Zero-cost Objective (minimize)")
    plt.title("Pareto Front of Evolution Search (Minimization View)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = Path(save_dir) / "pareto_front.png"
    plt.savefig(save_path, dpi=300)
    plt.close()


def search_candidates(args) -> list[dict[str, Any]]:
    """
    执行进化搜索，并返回 top-k 候选架构。

    进化搜索核心逻辑说明：
    1. 初始化种群：随机生成一批候选架构
    2. 个体评估：对每个候选架构计算两个目标
       - 目标1：zero-cost score（希望越大越好）
       - 目标2：参数量 params_mb（希望越小越好）
    3. 非支配排序：根据多目标优化思想，将个体划分为不同帕累托层
    4. 交叉变异：从优秀个体中产生新个体，形成下一代种群
    5. 迭代进化：重复多代，逐步逼近帕累托最优前沿
    6. 输出结果：保存 top-k 候选架构，并绘制帕累托前沿图
    """
    args_ns = _normalize_args(args)
    _validate_args(args_ns)

    metric_choices = _available_metrics()
    if args_ns.metric not in metric_choices:
        raise ValueError(f"不支持的 metric: {args_ns.metric}，可选: {metric_choices}")

    args_ns.save_dir = create_exp_dir(stage="search")
    logger = setup_logger(name="search", save_dir=args_ns.save_dir, level="INFO")

    # 输出搜索开始摘要
    logger.info("=" * 60)
    logger.info("EvoHCell-NAS 进化搜索开始")
    logger.info("=" * 60)
    logger.info(f"进化代数: {args_ns.generations}")
    logger.info(f"种群大小: {args_ns.population_size}")
    logger.info(f"Zero-cost 指标: {args_ns.metric}")
    logger.info(f"网络层数: {args_ns.layers}")
    logger.info(f"层级交叉概率: {args_ns.pc_layer}")
    logger.info(f"层级变异概率: {args_ns.pm_layer}")
    logger.info(f"随机种子: {args_ns.seed}")
    logger.info(f"最大化 score: {args_ns.maximize_score}")
    logger.info("=" * 60)

    _set_random_seed(args_ns.seed)

    search_config = _build_search_config(args_ns)

    # 延迟导入，避免模块导入阶段产生额外副作用
    from genetic.NSGA_II import run_nsga2

    # ------------------------------
    # 进化搜索主过程
    # run_nsga2 内部通常会完成：
    # 1. 初始化种群
    # 2. 计算每个个体的多目标 fitness
    # 3. 非支配排序与拥挤距离选择
    # 4. 交叉与变异，生成新一代种群
    # 5. 迭代多代后返回最终种群和帕累托分层结果
    # ------------------------------
    final_pop, fronts = run_nsga2(search_config, logger)

    maximize_score = bool(search_config.get("maximize_score", True))
    ranked_indices = _rank_candidate_indices(fronts, len(final_pop.individuals))
    top_indices = ranked_indices[: args_ns.top_k]

    candidates: list[dict[str, Any]] = []
    for cid, idx in enumerate(top_indices):
        ind = final_pop.individuals[idx]

        genotype_json_list = []
        for g in ind.genotype:
            g_json = _genotype_to_json_dict(g)
            genotype_json_list.append(g_json)

        # 记录当前个体属于第几层帕累托前沿
        front_rank = None
        for fid, front in enumerate(fronts):
            if idx in front:
                front_rank = fid
                break

        candidate = {
            "id": cid,
            "front_rank": front_rank,
            "genotype_list": genotype_json_list,
            "zero_cost_score": _restore_zero_cost_score(ind.fitness, maximize_score),
            "zero_cost_objective": float(ind.fitness[0]),  # 内部最小化目标，用于 Pareto 绘图
            "params_mb": float(ind.fitness[1]),
            "fitness": [float(x) for x in ind.fitness],
        }
        candidates.append(candidate)

    # 这里按 Pareto 排名顺序输出，不再额外按单目标重排
    for new_id, candidate in enumerate(candidates):
        candidate["id"] = new_id

    result = {
        "meta": {
            "save_dir": args_ns.save_dir,
            "metric": args_ns.metric,
            "seed": args_ns.seed,
            "generations": args_ns.generations,
            "population_size": args_ns.population_size,
            "top_k": args_ns.top_k,
            "layers": args_ns.layers,
            "pc_layer": args_ns.pc_layer,
            "pm_layer": args_ns.pm_layer,
            "maximize_score": args_ns.maximize_score,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "candidates": candidates,
    }

    output_path = Path(args_ns.save_dir) / "top_candidates.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 生成人工可读的文本文件
    readable_path = Path(args_ns.save_dir) / "top_candidates_readable.txt"
    with open(readable_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("EvoHCell-NAS 搜索结果 - 可读格式\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"搜索配置:\n")
        f.write(f"  - 指标: {args_ns.metric}\n")
        f.write(f"  - 进化代数: {args_ns.generations}\n")
        f.write(f"  - 种群大小: {args_ns.population_size}\n")
        f.write(f"  - 网络层数: {args_ns.layers}\n")
        f.write(f"  - 随机种子: {args_ns.seed}\n\n")

        for candidate in candidates:
            f.write("=" * 80 + "\n")
            f.write(f"候选架构 #{candidate['id']}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Pareto 前沿层级: {candidate['front_rank']}\n")
            f.write(f"Zero-cost Score: {candidate['zero_cost_score']:.6f}\n")
            f.write(f"参数量 (MB): {candidate['params_mb']:.4f}\n\n")

            for layer_idx, genotype_dict in enumerate(candidate['genotype_list']):
                f.write(f"Layer {layer_idx}:\n")
                f.write(f"  normal: {_edges_to_inline_str(genotype_dict['normal'])}\n")
                f.write(f"  reduce: {_edges_to_inline_str(genotype_dict['reduce'])}\n")
            f.write("\n")

    # 为每个候选单独导出 JSON 文件
    for candidate in candidates:
        candidate_file = Path(args_ns.save_dir) / f"candidate_{candidate['id']}.json"
        candidate_data = {
            "id": candidate["id"],
            "front_rank": candidate["front_rank"],
            "genotype_list": candidate["genotype_list"],
            "zero_cost_score": candidate["zero_cost_score"],
            "params_mb": candidate["params_mb"],
            "meta": {
                "metric": args_ns.metric,
                "layers": args_ns.layers,
                "seed": args_ns.seed,
            }
        }
        with open(candidate_file, "w", encoding="utf-8") as f:
            json.dump(candidate_data, f, ensure_ascii=False, indent=2)

    _plot_pareto_front(
        final_pop=final_pop,
        fronts=fronts,
        save_dir=args_ns.save_dir,
        maximize_score=maximize_score,
    )

    # 输出最终摘要
    first_front_size = len(fronts[0]) if fronts else 0
    pareto_path = Path(args_ns.save_dir) / "pareto_front.png"
    logger.info("=" * 60)
    logger.info("搜索完成，最终摘要")
    logger.info(f"最终第一帕累托前沿大小: {first_front_size}")
    logger.info(f"top_k 实际导出候选数: {len(candidates)}")
    logger.info("=" * 60)

    return candidates, args_ns.save_dir


def build_parser() -> argparse.ArgumentParser:
    metric_choices = _available_metrics()

    parser = argparse.ArgumentParser("search_cifar10")

    # =========================
    # 数据与训练基础参数
    # =========================
    parser.add_argument('--data', type=str, default='./data', help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='初始学习率')
    parser.add_argument('--min_learning_rate', type=float, default=1e-3, help='最小学习率')
    parser.add_argument('--momentum', type=float, default=0.9, help='动量')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='权重衰减')
    parser.add_argument('--report_freq', type=float, default=50, help='日志打印频率')

    # =========================
    # 搜索相关参数
    # =========================
    parser.add_argument('--generations', type=int, default=30, help='进化代数')
    parser.add_argument('--population_size', type=int, default=50, help='种群大小')
    parser.add_argument('--top_k', type=int, default=5, help='返回 top-k 候选架构')
    parser.add_argument('--pc_layer', type=float, default=0.5, help='层级交叉概率')
    parser.add_argument('--pm_layer', type=float, default=0.2, help='层级变异概率')
    parser.add_argument('--layers', type=int, default=20, help='网络层数')  # 20
    parser.add_argument('--init_channels', type=int, default=36, help='search阶段的初始通道数')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    # grad_norm, synflow
    parser.add_argument('--metric', type=str, default='grad_norm', choices=metric_choices, help='zero-cost 指标')
    parser.add_argument('--maximize_score', action='store_true', default=True, help='是否最大化 zero-cost score')
    parser.add_argument('--save_dir', type=str, default=None, help='结果保存目录')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别')

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    candidates, save_dir = search_candidates(args)

    print(f"搜索完成，返回候选数量: {len(candidates)}")
    print(f"候选架构已保存到: {Path(save_dir) / 'top_candidates.json'}")
    print(f"帕累托前沿图已保存到: {Path(save_dir) / 'pareto_front.png'}")


if __name__ == "__main__":
    main()
