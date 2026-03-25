"""测试变异算子能产生不同个体"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_mutation_produces_different_individual():
    """变异概率设为1.0时，变异后的个体应与原始个体不同"""
    from genetic.population import Population
    from genetic.crossover_and_mutation import gaussian_mutation
    from genetic.config_loader import get_evolution_config

    config = get_evolution_config()
    config["pop_size"] = 1
    config["pm_layer"] = 1.0  # 强制每层都变异
    config["pm_edge"] = 1.0   # 强制每条边都变异
    config["init_population_path"] = str(Path(__file__).parent.parent / "genetic" / "init_population.txt")

    pop = Population(config)
    pop.initialize()

    original = pop.individuals[0]
    mutated = gaussian_mutation(original, config)

    # 至少有一层的基因型应该不同
    different = False
    for g_orig, g_mut in zip(original.genotype, mutated.genotype):
        if g_orig.normal != g_mut.normal or g_orig.reduce != g_mut.reduce:
            different = True
            break

    assert different, "变异概率为1.0时，变异算子应产生不同的个体"


def test_mutation_resets_fitness():
    """变异后的个体适应度应被重置为inf"""
    from genetic.population import Population
    from genetic.crossover_and_mutation import gaussian_mutation
    from genetic.config_loader import get_evolution_config

    config = get_evolution_config()
    config["pop_size"] = 1
    config["pm_layer"] = 1.0
    config["pm_edge"] = 1.0
    config["init_population_path"] = str(Path(__file__).parent.parent / "genetic" / "init_population.txt")

    pop = Population(config)
    pop.initialize()

    original = pop.individuals[0]
    # 模拟已评估的适应度
    original.fitness = [0.5, 100.0]

    mutated = gaussian_mutation(original, config)

    # 变异后适应度应重置
    assert all(f == float("inf") for f in mutated.fitness), "变异后适应度应重置为inf"


def test_zero_mutation_preserves_individual():
    """变异概率为0时，个体应保持不变"""
    from genetic.population import Population
    from genetic.crossover_and_mutation import gaussian_mutation
    from genetic.config_loader import get_evolution_config

    config = get_evolution_config()
    config["pop_size"] = 1
    config["pm_layer"] = 0.0  # 不变异
    config["pm_edge"] = 0.0
    config["init_population_path"] = str(Path(__file__).parent.parent / "genetic" / "init_population.txt")

    pop = Population(config)
    pop.initialize()

    original = pop.individuals[0]
    mutated = gaussian_mutation(original, config)

    # 所有层的基因型应相同
    for g_orig, g_mut in zip(original.genotype, mutated.genotype):
        assert g_orig.normal == g_mut.normal, "变异概率为0时，normal cell不应改变"
        assert g_orig.reduce == g_mut.reduce, "变异概率为0时，reduce cell不应改变"


if __name__ == "__main__":
    test_mutation_produces_different_individual()
    print("通过: 变异算子产生不同个体")

    test_mutation_resets_fitness()
    print("通过: 变异后适应度重置")

    test_zero_mutation_preserves_individual()
    print("通过: 零变异概率保持个体不变")

    print("\n所有变异算子测试通过!")
