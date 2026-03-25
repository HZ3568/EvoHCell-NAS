"""Environment selection strategies.

Currently provides NSGA-II environment selection that combines parent and
offspring populations, performs non-dominated sorting and crowding-distance
based truncation to maintain population size.
"""

from __future__ import annotations
from typing import Sequence

from .population import Population, Individual
from .NSGA_II import NSGAII


def nsga2_environment_selection(pop: Population, offspring: Sequence[Individual]) -> None:
    """Apply NSGA-II environment selection to update the population in place."""
    nsga = NSGAII()
    combined = pop.individuals + list(offspring)
    next_gen = nsga.select_next_generation(combined, pop.pop_size)
    pop.individuals = next_gen
    pop.front = nsga.fast_nondominated_sort(pop.individuals)
    for front in pop.front:
        nsga.crowding_distance(front, pop.individuals)