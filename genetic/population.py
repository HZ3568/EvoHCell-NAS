from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Sequence, Optional, Dict, Any
import random
from pathlib import Path
import re

from darts.genotypes import Genotype


def load_genotype_pool(file_path: str) -> List[Genotype]:
    """
    从文件中加载 genotype。

    支持两种格式：
    1. Genotype(...)
    2. Genotype(...) valid_loss:27.4620
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"init population file not found: {file_path}")

    genotypes: List[Genotype] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue

            try:
                # 提取前面的 Genotype(...)，忽略后面的 valid_loss:...
                match = re.match(r"^(Genotype\(.*\))\s*(?:valid_loss:\s*[-+]?\d*\.?\d+)?\s*$", text)
                if not match:
                    raise ValueError(f"Line {line_num}: Invalid format")

                genotype_str = match.group(1)

                # 限制命名空间，尽量安全
                namespace = {
                    "Genotype": Genotype,
                    "range": range,
                    "list": list,
                }

                genotype = eval(genotype_str, namespace, {})
                genotypes.append(genotype)

            except Exception as e:
                raise ValueError(
                    f"Line {line_num}: Failed to parse genotype: {e}\n"
                    f"Line content: {text}"
                )

    if not genotypes:
        raise ValueError(f"no genotype parsed from: {file_path}")

    return genotypes


@dataclass
class Individual:
    genotype: Any
    fitness: List[float]

    # NSGA-II bookkeeping
    Sp: List[int] = None
    Np: int = 0
    p_rank: Optional[int] = None
    crowd_distance: float = 0.0

    def __post_init__(self):
        if self.Sp is None:
            self.Sp = []

    def get_F_value(self) -> List[float]:
        return self.fitness

    def copy(self) -> "Individual":
        # 对当前 genotype 结构做一个尽量安全的浅复制
        if isinstance(self.genotype, list):
            cloned_genotype = self.genotype[:]
        else:
            cloned_genotype = self.genotype

        new_one = Individual(cloned_genotype, self.fitness[:])
        new_one.Sp = self.Sp[:]
        new_one.Np = self.Np
        new_one.p_rank = self.p_rank
        new_one.crowd_distance = self.crowd_distance
        return new_one


class Population:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.copy()
        self.pop_size: int = int(self.config.get("pop_size", 50))
        self.var_bounds: Optional[Sequence[Sequence[float]]] = self.config.get("var_bounds")
        self.num_vars: int = int(self.config.get("num_vars", len(self.var_bounds) if self.var_bounds else 1))
        self.objectives: int = int(self.config.get("objectives", 1))
        self.layers: int = int(self.config.get("layers", 8))
        self.init_population_path: Optional[str] = self.config.get("init_population_path")
        self.init_strategy: Optional[Callable[[], Any]] = self.config.get("init_strategy")

        seed = self.config.get("seed")
        if seed is not None:
            random.seed(seed)

        self.individuals: List[Individual] = []
        self.front: List[List[int]] = []

    def _default_init(self) -> List[float]:
        if self.var_bounds:
            return [random.uniform(lo, hi) for (lo, hi) in self.var_bounds]
        return [random.random() for _ in range(self.num_vars)]

    def _build_genotype_list_population(self) -> List[Individual]:
        """
        根据 init_population.txt 中的 genotype 逐行构造初始种群。
        文件中有多少个 genotype，就生成多少个 individual。
        """
        if not self.init_population_path:
            raise ValueError("config must provide init_population_path for genotype_list initialization")

        genotype_pool = load_genotype_pool(self.init_population_path)
        individuals: List[Individual] = []

        for g in genotype_pool:
            genotype_list = [g for _ in range(self.layers)]
            individuals.append(
                Individual(
                    genotype=genotype_list,
                    fitness=[float("inf")] * self.objectives
                )
            )

        return individuals

    def initialize(self):
        """
        初始化种群。
        """
        if self.init_strategy is not None:
            init_fn = self.init_strategy
            self.individuals = [
                Individual(genotype=init_fn(), fitness=[float("inf")] * self.objectives)
                for _ in range(self.pop_size)
            ]
            return

        if self.init_population_path:
            self.individuals = self._build_genotype_list_population()
            self.pop_size = len(self.individuals)
            return

        self.individuals = [
            Individual(genotype=self._default_init(), fitness=[float("inf")] * self.objectives)
            for _ in range(self.pop_size)
        ]

    def extend(self, new_individuals: Sequence[Individual]):
        self.individuals.extend(new_individuals)

    def top_n_by_crowding(self, indices: Sequence[int], n: int) -> List[int]:
        return sorted(indices, key=lambda i: self.individuals[i].crowd_distance, reverse=True)[:n]

    def size(self) -> int:
        return len(self.individuals)

    def __len__(self):
        return self.size()