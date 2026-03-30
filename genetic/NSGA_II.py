"""NSGA-II核心算法和进化循环

实现非支配排序、拥挤距离计算和NSGA-II工作流程。
设计用于与Population、Individual、评估器和变异算子集成。
"""

from __future__ import annotations
from typing import List, Sequence, Dict, Any, Tuple, Optional
import logging
import random
from pathlib import Path

from .population import Population, Individual
from .evaluate import Evaluator
from .crossover_and_mutation import layerwise_crossover


def _pop_stats(individuals: Sequence[Individual]) -> Dict[str, Any]:
    """统计种群两个目标的 min/mean/max。"""
    if not individuals:
        return {}
    obj0 = [float(ind.fitness[0]) for ind in individuals]
    obj1 = [float(ind.fitness[1]) for ind in individuals]
    return {
        "zero_cost": {"min": min(obj0), "mean": sum(obj0) / len(obj0), "max": max(obj0)},
        "params_mb": {"min": min(obj1), "mean": sum(obj1) / len(obj1), "max": max(obj1)},
    }


def _front_representatives(
    front_indices: List[int], individuals: Sequence[Individual]
) -> Dict[str, Any]:
    """从第一前沿中提取代表性个体（params_mb 最小 / zero-cost 最小）。"""
    if not front_indices:
        return {}
    front_inds = [(i, individuals[i]) for i in front_indices]
    best_params = min(front_inds, key=lambda x: float(x[1].fitness[1]))
    best_score = min(front_inds, key=lambda x: float(x[1].fitness[0]))
    return {
        "best_params": {"idx": best_params[0], "params_mb": float(best_params[1].fitness[1]), "zero_cost_obj": float(best_params[1].fitness[0])},
        "best_score": {"idx": best_score[0], "params_mb": float(best_score[1].fitness[1]), "zero_cost_obj": float(best_score[1].fitness[0])},
    }


def _log_gen_summary(
    logger: logging.Logger,
    gen_idx: int,
    generations: int,
    fronts: List[List[int]],
    individuals: Sequence[Individual],
) -> None:
    """输出一代的简洁摘要日志。"""
    first_front_size = len(fronts[0]) if fronts else 0
    stats = _pop_stats(individuals)
    reps = _front_representatives(fronts[0] if fronts else [], individuals)

    zc = stats.get("zero_cost", {})
    pm = stats.get("params_mb", {})
    logger.info(
        f"[Gen {gen_idx + 1:>3}/{generations}] "
        f"前沿大小={first_front_size} | "
        f"zero_cost obj: min={zc.get('min', 0):.4f} mean={zc.get('mean', 0):.4f} max={zc.get('max', 0):.4f} | "
        f"params_mb: min={pm.get('min', 0):.2f} mean={pm.get('mean', 0):.2f} max={pm.get('max', 0):.2f}"
    )
    if reps:
        bp = reps["best_params"]
        bs = reps["best_score"]
        logger.info(
            f"  代表个体 -> "
            f"params最小: params={bp['params_mb']:.2f}MB obj={bp['zero_cost_obj']:.4f} | "
            f"score最优: params={bs['params_mb']:.2f}MB obj={bs['zero_cost_obj']:.4f}"
        )


class NSGAII:
    """NSGA-II排序和选择工具类"""

    def is_dominate(self, a: Individual, b: Individual) -> bool:
        """判断个体a是否支配个体b（最小化目标）"""
        a_f = a.get_F_value()
        b_f = b.get_F_value()
        better = False
        for av, bv in zip(a_f, b_f):
            if av > bv:
                return False
            if av < bv:
                better = True
        return better

    def compute_first_front(self, individuals: Sequence[Individual]) -> List[int]:
        F1: List[int] = []
        for j, p in enumerate(individuals):
            p.Sp = []
            p.Np = 0
            for i, q in enumerate(individuals):
                if j == i:
                    continue
                if self.is_dominate(p, q):
                    if i not in p.Sp:
                        p.Sp.append(i)
                elif self.is_dominate(q, p):
                    p.Np += 1
            if p.Np == 0:
                p.p_rank = 1
                F1.append(j)
        return F1

    def fast_nondominated_sort(self, individuals: Sequence[Individual]) -> List[List[int]]:
        F: List[List[int]] = []
        i = 1
        F1 = self.compute_first_front(individuals)
        while len(F1) != 0:
            F.append(F1)
            Q: List[int] = []
            for pi in F1:
                p = individuals[pi]
                for q in p.Sp:
                    one_q = individuals[q]
                    one_q.Np -= 1
                    if one_q.Np == 0:
                        one_q.p_rank = i + 1
                        Q.append(q)
            i += 1
            F1 = Q
        return F

    def crowding_distance(self, front: List[int], individuals: Sequence[Individual]) -> None:
        """计算前沿中个体的拥挤距离

        使用有限大值代替inf避免数值问题，分母添加epsilon保证数值稳定性

        Args:
            front: 前沿中个体的索引列表
            individuals: 所有个体的序列
        """
        if not front:
            return

        f_num = len(individuals[front[0]].get_F_value())

        # 重置距离并计算目标值的最大最小值
        f_max = individuals[front[0]].get_F_value()[:]
        f_min = individuals[front[0]].get_F_value()[:]
        for idx in front:
            ind = individuals[idx]
            ind.crowd_distance = 0.0
            for m in range(f_num):
                val = ind.get_F_value()[m]
                if val > f_max[m]:
                    f_max[m] = val
                if val < f_min[m]:
                    f_min[m] = val

        # 边界个体使用大有限值代替inf
        LARGE_VALUE = 1e9

        # 按每个目标排序并分配距离
        for m in range(f_num):
            front_sorted = sorted(front, key=lambda i: individuals[i].get_F_value()[m])

            # 边界个体获得大有限值
            individuals[front_sorted[0]].crowd_distance = LARGE_VALUE
            individuals[front_sorted[-1]].crowd_distance = LARGE_VALUE

            # 内部个体获得归一化距离
            for k in range(1, len(front_sorted) - 1):
                next_val = individuals[front_sorted[k + 1]].get_F_value()[m]
                prev_val = individuals[front_sorted[k - 1]].get_F_value()[m]

                # 添加epsilon避免除零
                denom = max(f_max[m] - f_min[m], 1e-10)
                delta = (next_val - prev_val) / denom
                individuals[front_sorted[k]].crowd_distance += delta

    def select_next_generation(self, combined: Sequence[Individual], pop_size: int) -> List[Individual]:
        fronts = self.fast_nondominated_sort(combined)
        selected: List[Individual] = []
        for front in fronts:
            if len(selected) + len(front) <= pop_size:
                # 整个前沿都能放入
                self.crowding_distance(front, combined)
                selected.extend([combined[i] for i in front])
            else:
                # 需要根据拥挤距离选择
                self.crowding_distance(front, combined)
                # 按拥挤距离降序排序
                front_sorted = sorted(front, key=lambda i: combined[i].crowd_distance, reverse=True)
                needed = pop_size - len(selected)
                selected.extend([combined[i] for i in front_sorted[:needed]])
                break
        return selected

    def update_rank_and_crowding(self, individuals: Sequence[Individual]) -> None:
        """更新种群中所有个体的 p_rank 和 crowd_distance

        在父代选择前调用，确保锦标赛选择有正确的排序依据

        Args:
            individuals: 当前种群的所有个体
        """
        fronts = self.fast_nondominated_sort(individuals)
        for front in fronts:
            self.crowding_distance(front, individuals)

    def better(self, a: Individual, b: Individual) -> bool:
        """判断个体 a 是否优于个体 b（用于锦标赛选择）

        比较规则：
        1. p_rank 更小的更优（非支配等级越小越好）
        2. 如果 p_rank 相同，crowd_distance 更大的更优（拥挤距离越大越好）

        Args:
            a: 个体 a
            b: 个体 b

        Returns:
            True 如果 a 优于 b，否则 False
        """
        if a.p_rank < b.p_rank:
            return True
        elif a.p_rank > b.p_rank:
            return False
        else:
            # p_rank 相同，比较拥挤距离
            return a.crowd_distance > b.crowd_distance

    def tournament_select_one(self, individuals: Sequence[Individual]) -> Individual:
        """二元锦标赛选择一个个体

        随机选择两个个体，返回更优的那个

        Args:
            individuals: 候选个体序列

        Returns:
            选中的个体
        """
        a, b = random.sample(list(individuals), 2)
        return a if self.better(a, b) else b


def run_nsga2(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Tuple[Population, List[List[int]]]:
    """运行 NSGA-II 进化搜索。

    Args:
        config: 搜索配置字典
        logger: 日志记录器，如果为 None 则使用默认 logger

    Returns:
        (最终种群, 帕累托前沿列表)
    """
    if logger is None:
        logger = logging.getLogger("EvoHCell-NAS")

    cfg = config.copy()
    if "init_population_path" not in cfg:
        cfg["init_population_path"] = str(Path(__file__).with_name("init_population.txt"))
    init_pop_size = int(cfg.get("init_pop_size", cfg.get("population_size", 20)))
    max_population_size = int(cfg.get("population_size", 20))
    generations = int(cfg.get("generations", 20))
    parent_pool_size = int(cfg.get("parent_pool_size", 10))
    crossover_rounds = int(cfg.get("crossover_rounds", 10))
    cfg["pop_size"] = init_pop_size
    cfg["objectives"] = 2

    logger.info("初始化种群...")
    pop = Population(cfg)
    pop.initialize()
    evaluator = Evaluator(cfg)
    evaluator.evaluate_population(pop)

    nsga = NSGAII()
    fronts = nsga.fast_nondominated_sort(pop.individuals)
    pop.front = fronts

    # 输出初始化信息
    first_front_size = len(fronts[0]) if fronts else 0
    logger.info(f"初始种群大小: {len(pop.individuals)}")
    logger.info(f"初始第一前沿大小: {first_front_size}")
    logger.debug(f"初始种群统计: {_pop_stats(pop.individuals)}")
    logger.info("-" * 60)

    def select_two_parents() -> Tuple[Individual, Individual]:
        """使用二元锦标赛选择两个不同的父代个体

        Returns:
            两个不同的父代个体
        """
        p1 = nsga.tournament_select_one(pop.individuals)
        # 避免选到同一个父代，最多重试 10 次
        for _ in range(10):
            p2 = nsga.tournament_select_one(pop.individuals)
            if p2 is not p1:
                return p1, p2
        # 如果重试失败，直接随机选一个不同的
        candidates = [ind for ind in pop.individuals if ind is not p1]
        if candidates:
            p2 = random.choice(candidates)
        else:
            p2 = p1  # 极端情况：种群只有一个个体
        return p1, p2

    for gen_idx in range(generations):
        logger.debug(f"开始第 {gen_idx + 1} 代交叉变异...")

        # 在父代选择前更新所有个体的 rank 和 crowding distance
        nsga.update_rank_and_crowding(pop.individuals)

        offsprings: List[Individual] = []
        for _ in range(crossover_rounds):
            p1, p2 = select_two_parents()
            c1, c2 = layerwise_crossover(p1, p2, cfg)
            offsprings.append(c1)
            offsprings.append(c2)

        logger.debug(f"评估 {len(offsprings)} 个新个体...")
        for child in offsprings:
            evaluator.evaluate_individual(child)
        combined = pop.individuals + offsprings

        logger.debug(f"选择下一代种群（当前合并种群大小: {len(combined)}）...")
        if len(combined) > max_population_size:
            pop.individuals = nsga.select_next_generation(combined, max_population_size)
        else:
            pop.individuals = combined

        fronts = nsga.fast_nondominated_sort(pop.individuals)
        pop.front = fronts

        # 输出每一代的摘要
        _log_gen_summary(logger, gen_idx, generations, fronts, pop.individuals)

    logger.info("-" * 60)
    logger.info("进化循环完成")

    return pop, pop.front


if __name__ == "__main__":
    # 示例配置，实际使用时应从 search.py 传入
    config = {
        "data": "./data",
        "batch_size": 96,
        "generations": 20,
        "population_size": 20,
        "layers": 20,
        "metric": "synflow",
        "maximize_score": True,
        "seed": 0,
    }
    final_pop, fronts = run_nsga2(config)
    print(f"总前沿数: {len(fronts)}; 第一前沿大小: {len(fronts[0])}")
    first_front_inds = [final_pop.individuals[i] for i in fronts[0]]
    for ind in first_front_inds[:5]:
        print("genotype=", ind.genotype, "fitness=", ind.fitness)

