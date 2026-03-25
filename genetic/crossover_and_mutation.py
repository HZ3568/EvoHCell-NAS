"""Crossover operators for genotype-list based evolutionary algorithms."""

from __future__ import annotations
from typing import Tuple, Dict, Any
import random

try:
    from genetic.population import Individual
except ImportError:
    from population import Individual


def layerwise_crossover(p1: Individual, p2: Individual, config: Dict[str, Any]) -> Tuple[Individual, Individual]:
    p_layer = float(config.get("pc_layer", 0.5))
    g1 = p1.genotype[:]
    g2 = p2.genotype[:]
    if len(g1) != len(g2):
        raise ValueError("parents must have same genotype_list length")
    c1 = g1[:]
    c2 = g2[:]
    for i in range(len(g1)):
        if random.random() < p_layer:
            c1[i], c2[i] = c2[i], c1[i]
    return Individual(c1, [float("inf")] * len(p1.fitness)), Individual(c2, [float("inf")] * len(p2.fitness))


def arithmetic_crossover(p1: Individual, p2: Individual, config: Dict[str, Any], var_bounds=None) -> Tuple[Individual, Individual]:
    return layerwise_crossover(p1, p2, config)


def gaussian_mutation(ind: Individual, config: Dict[str, Any], var_bounds=None) -> Individual:
    """Mutate an individual by randomly replacing operations in genotypes.

    Args:
        ind: Individual to mutate
        config: Configuration dict containing pm_layer and pm_edge
        var_bounds: Not used, kept for compatibility

    Returns:
        Mutated individual with reset fitness
    """
    from darts.genotypes import PRIMITIVES, Genotype

    pm_layer = float(config.get("pm_layer", 0.15))
    pm_edge = float(config.get("pm_edge", 0.3))

    mutated_genotype_list = []
    mutated = False

    for genotype in ind.genotype:
        # Decide if this layer should be mutated
        if random.random() < pm_layer:
            # Mutate normal cell
            new_normal = list(genotype.normal)
            for i in range(len(new_normal)):
                if random.random() < pm_edge:
                    op, node_idx = new_normal[i]
                    # Replace operation with random choice from PRIMITIVES
                    new_op = random.choice(PRIMITIVES)
                    new_normal[i] = (new_op, node_idx)
                    mutated = True

            # Mutate reduce cell
            new_reduce = list(genotype.reduce)
            for i in range(len(new_reduce)):
                if random.random() < pm_edge:
                    op, node_idx = new_reduce[i]
                    # Replace operation with random choice from PRIMITIVES
                    new_op = random.choice(PRIMITIVES)
                    new_reduce[i] = (new_op, node_idx)
                    mutated = True

            # Create new genotype with mutated cells (keep concat unchanged)
            new_genotype = Genotype(
                normal=new_normal,
                normal_concat=genotype.normal_concat,
                reduce=new_reduce,
                reduce_concat=genotype.reduce_concat
            )
            mutated_genotype_list.append(new_genotype)
        else:
            # Keep original genotype for this layer
            mutated_genotype_list.append(genotype)

    # Reset fitness if mutation occurred
    if mutated:
        return Individual(mutated_genotype_list, [float("inf")] * len(ind.fitness))
    else:
        # If no mutation occurred, return copy
        return ind.copy()
