"""Crossover operators for genotype-list based evolutionary algorithms."""

from __future__ import annotations

import math
import random
import re
from typing import Tuple, Dict, Any, List
from .population import Individual
from darts.genotypes import Genotype
from pathlib import Path


def crossover(p1: Individual, p2: Individual, config: Dict[str, Any]) -> Tuple[Individual, Individual]:
    pc_layer = float(config.get("pc_layer", 0.5))
    g1 = p1.genotype[:]
    g2 = p2.genotype[:]
    if len(g1) != len(g2):
        raise ValueError("parents must have same genotype_list length")
    c1 = g1[:]
    c2 = g2[:]
    for i in range(len(g1)):
        if random.random() < pc_layer:
            c1[i], c2[i] = c2[i], c1[i]
    return Individual(c1, [float("inf")] * len(p1.fitness)), Individual(c2, [float("inf")] * len(p2.fitness))


def softmax_sample_by_loss(candidates: List, losses: List[float], temperature: float):
    """
    根据 valid_loss 进行加权采样：loss 越小，采样概率越大。
    p ~ exp(-loss / temperature)
    """
    weights = [math.exp(-(loss - min(losses)) / temperature) for loss in losses]
    return random.choices(candidates, weights=weights, k=1)[0]


def load_arch_pool() -> Tuple[List[Genotype], List[float]]:
    txt_path = Path(__file__).with_name("init_population.txt")

    arch_pool: List[Genotype] = []
    arch_pool_losses: List[float] = []

    pattern = re.compile(r"^(Genotype\(.*\))\s+valid_loss:([0-9]*\.?[0-9]+)\s*$")

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            match = pattern.match(line)
            genotype_str = match.group(1)
            genotype = eval(genotype_str, {"Genotype": Genotype})
            valid_loss = float(match.group(2))
            arch_pool.append(genotype)
            arch_pool_losses.append(valid_loss)

    return arch_pool, arch_pool_losses


def mutation(individual: Individual, config: Dict[str, Any]) -> Individual:
    """
    对 genotype_list 中的每个位置，以 pm_layer 的概率，从架构池中按 valid_loss 加权采样一个新架构进行替换。
    """
    pm_layer = float(config.get("pm_layer", 0.02))
    temperature = float(config.get("mutation_temperature", 1.0))
    arch_pool, arch_pool_losses = load_arch_pool()

    g = individual.genotype[:]
    mutated = False

    for i in range(len(g)):
        if random.random() < pm_layer:
            g[i] = softmax_sample_by_loss(
                arch_pool,
                arch_pool_losses,
                temperature=temperature
            )
            mutated = True
    if mutated:
        return Individual(g, [float("inf")] * len(individual.fitness))
    return individual.copy()


if __name__ == "__main__":
    pass
