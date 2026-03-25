



"""Population and Individual definitions for evolutionary algorithms.

This module provides minimal yet extensible classes for representing individuals
and populations. It is designed to work with both single-objective GA and
multi-objective NSGA-II, following a modular structure.

Key features:
- Real-valued genotype representation (vector of floats)
- Fitness stored as list of objective values (minimization by default)
- Attributes required by NSGA-II: `Sp`, `Np`, `p_rank`, `crowd_distance`
- Configurable population initialization within value bounds

You can extend these classes to support different encodings or constraints.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Sequence, Optional, Dict, Any
import random
from pathlib import Path

from darts.genotypes import Genotype


def load_genotype_pool(file_path: str) -> List[Genotype]:
    """Load genotypes from file using safe parsing.

    Args:
        file_path: Path to file containing genotype definitions

    Returns:
        List of parsed Genotype objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If no valid genotypes found or parsing fails
    """
    import ast
    import re

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
                # Safe parsing using ast.literal_eval for the data structures
                # Extract the Genotype constructor call
                match = re.match(r'Genotype\((.*)\)', text)
                if not match:
                    raise ValueError(f"Line {line_num}: Invalid genotype format")

                # Parse the arguments
                args_str = match.group(1)

                # Use ast.literal_eval to safely parse the arguments
                # We need to handle range() specially
                args_str_safe = args_str.replace('range(', 'list(range(')

                # Create a safe namespace with only Genotype
                namespace = {'Genotype': Genotype, 'range': range, 'list': list}

                # Parse using eval with restricted namespace (safer than unrestricted eval)
                # Note: This still uses eval but with very restricted namespace
                genotype = eval(f'Genotype({args_str})', namespace, {})
                genotypes.append(genotype)

            except Exception as e:
                raise ValueError(f"Line {line_num}: Failed to parse genotype: {e}\nLine content: {text}")

    if len(genotypes) == 0:
        raise ValueError(f"no genotype parsed from: {file_path}")

    return genotypes


@dataclass
class Individual:
    """An individual in the population.

    Attributes:
    - genotype: List of floats representing solution variables.
    - fitness: List of objective values (lower is better by default).
    - Sp: Indices of individuals dominated by this one (for NSGA-II).
    - Np: Domination count (number of individuals that dominate this one).
    - p_rank: Pareto front rank (for NSGA-II).
    - crowd_distance: Crowding distance (for NSGA-II selection).
    """

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
        """Return the fitness (objective values)."""
        return self.fitness

    def copy(self) -> "Individual":
        """Create a shallow copy suitable for GA operations."""
        cloned_genotype = self.genotype[:]
        new_one = Individual(cloned_genotype, self.fitness[:])
        new_one.Sp = self.Sp[:]
        new_one.Np = self.Np
        new_one.p_rank = self.p_rank
        new_one.crowd_distance = self.crowd_distance
        return new_one


class Population:
    """A collection of individuals with utilities for initialization.

    Parameters (config):
    - `pop_size`: Number of individuals in population.
    - `var_bounds`: Sequence of `(low, high)` bounds per variable.
    - `num_vars`: Number of decision variables (required if `var_bounds` is None).
    - `init_strategy`: Callable to generate a genotype; defaults to uniform random within bounds.
    - `objectives`: Number of objectives (defaults to 1).
    - `seed`: Optional seed for reproducibility.
    """

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
        self.front: List[List[int]] = []  # Pareto fronts as lists of indices

    def _default_init(self) -> List[float]:
        if self.var_bounds:
            return [random.uniform(lo, hi) for (lo, hi) in self.var_bounds]
        return [random.random() for _ in range(self.num_vars)]

    def _build_genotype_list_population(self) -> List[Individual]:
        if not self.init_population_path:
            raise ValueError("config must provide init_population_path for genotype_list initialization")
        genotype_pool = load_genotype_pool(self.init_population_path)
        individuals: List[Individual] = []
        for _ in range(self.pop_size):
            g = random.choice(genotype_pool)
            genotype_list = [g for _ in range(self.layers)]
            individuals.append(Individual(genotype=genotype_list, fitness=[float("inf")] * self.objectives))
        return individuals

    def initialize(self):
        """Initialize the population with random genotypes and zero fitness.

        Fitness values are initialized to `inf` for each objective.
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
            return

        self.individuals = [
            Individual(genotype=self._default_init(), fitness=[float("inf")] * self.objectives)
            for _ in range(self.pop_size)
        ]

    def extend(self, new_individuals: Sequence[Individual]):
        """Extend population with new individuals."""
        self.individuals.extend(new_individuals)

    def top_n_by_crowding(self, indices: Sequence[int], n: int) -> List[int]:
        """Return top `n` individuals by crowding distance from provided indices."""
        return sorted(indices, key=lambda i: self.individuals[i].crowd_distance, reverse=True)[:n]

    def size(self) -> int:
        return len(self.individuals)

    def __len__(self):
        return self.size()
