"""NSGA-II核心算法和进化循环

实现非支配排序、拥挤距离计算和NSGA-II工作流程。
设计用于与Population、Individual、评估器和变异算子集成。
"""

from __future__ import annotations
from typing import List, Sequence, Dict, Any, Tuple
import random
from pathlib import Path

try:
    from genetic.population import Population, Individual
    from genetic.evaluate import Evaluator
    from genetic.crossover_and_mutation import layerwise_crossover
    from experiments.config_loader import get_evolution_config
except ImportError:
    from population import Population, Individual
    from evaluate import Evaluator
    from crossover_and_mutation import layerwise_crossover
    from experiments.config_loader import get_evolution_config


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


def run_nsga2(config: Dict[str, Any]) -> Tuple[Population, List[List[int]]]:
    cfg = get_evolution_config()
    cfg.update(config)
    if "init_population_path" not in cfg:
        cfg["init_population_path"] = str(Path(__file__).with_name("init_population.txt"))
    init_pop_size = int(cfg["init_pop_size"])
    max_population_size = int(cfg["population_size"])
    generations = int(cfg["generations"])
    parent_pool_size = int(cfg["parent_pool_size"])
    crossover_rounds = int(cfg["crossover_rounds"])
    cfg["pop_size"] = init_pop_size
    pop = Population(cfg)
    pop.initialize()
    evaluator = Evaluator(cfg)
    evaluator.evaluate_population(pop)

    nsga = NSGAII()
    fronts = nsga.fast_nondominated_sort(pop.individuals)
    pop.front = fronts

    def score(ind: Individual) -> float:
        return float(sum(ind.fitness))

    def select_two_parents(gen_idx: int) -> Tuple[Individual, Individual]:
        if gen_idx == 0:
            return tuple(random.sample(pop.individuals, 2))
        ordered = sorted(pop.individuals, key=score)
        k = min(max(2, parent_pool_size), len(ordered))
        return tuple(random.sample(ordered[:k], 2))

    for gen_idx in range(generations):
        offsprings: List[Individual] = []
        for _ in range(crossover_rounds):
            p1, p2 = select_two_parents(gen_idx)
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

    return pop, pop.front


if __name__ == "__main__":
    config = get_evolution_config()
    final_pop, fronts = run_nsga2(config)
    print(f"总前沿数: {len(fronts)}; 第一前沿大小: {len(fronts[0])}")
    first_front_inds = [final_pop.individuals[i] for i in fronts[0]]
    for ind in first_front_inds[:5]:
        print("genotype=", ind.genotype, "fitness=", ind.fitness)

