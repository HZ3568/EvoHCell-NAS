"""测试拥挤距离计算的数值稳定性"""
import sys
import math
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# NSGA_II导入链会触发torch/torchvision，在不可用时mock掉
try:
    import torch
except (OSError, ImportError):
    _mock = unittest.mock.MagicMock()
    for mod in ["torch", "torch.nn", "torch.optim", "torch.hub",
                 "torch.utils", "torch.utils.data",
                 "torchvision", "torchvision.datasets",
                 "torchvision.transforms", "torchvision.extension",
                 "torchvision._internally_replaced_utils"]:
        sys.modules[mod] = _mock

from genetic.NSGA_II import NSGAII
from genetic.population import Individual


def test_identical_fitness_no_inf():
    """适应度完全相同的个体，拥挤距离不应出现inf"""
    individuals = [
        Individual([1, 2, 3], [1.0, 2.0]),
        Individual([4, 5, 6], [1.0, 2.0]),
        Individual([7, 8, 9], [1.0, 2.0]),
    ]

    nsga = NSGAII()
    nsga.crowding_distance([0, 1, 2], individuals)

    for ind in individuals:
        assert not math.isinf(ind.crowd_distance), \
            f"拥挤距离不应为inf，实际为 {ind.crowd_distance}"


def test_identical_fitness_no_nan():
    """适应度完全相同的个体，拥挤距离不应出现nan"""
    individuals = [
        Individual([1], [5.0, 5.0]),
        Individual([2], [5.0, 5.0]),
        Individual([3], [5.0, 5.0]),
    ]

    nsga = NSGAII()
    nsga.crowding_distance([0, 1, 2], individuals)

    for ind in individuals:
        assert not math.isnan(ind.crowd_distance), \
            f"拥挤距离不应为nan，实际为 {ind.crowd_distance}"


def test_boundary_individuals_get_large_distance():
    """前沿边界个体应获得较大的拥挤距离"""
    individuals = [
        Individual([1], [1.0, 10.0]),  # 目标1最小
        Individual([2], [5.0, 5.0]),   # 中间
        Individual([3], [10.0, 1.0]),  # 目标1最大
    ]

    nsga = NSGAII()
    nsga.crowding_distance([0, 1, 2], individuals)

    # 边界个体的拥挤距离应大于中间个体
    assert individuals[0].crowd_distance > individuals[1].crowd_distance, \
        "边界个体拥挤距离应大于中间个体"
    assert individuals[2].crowd_distance > individuals[1].crowd_distance, \
        "边界个体拥挤距离应大于中间个体"


def test_two_individuals():
    """只有两个个体时不应崩溃"""
    individuals = [
        Individual([1], [1.0, 2.0]),
        Individual([2], [3.0, 4.0]),
    ]

    nsga = NSGAII()
    nsga.crowding_distance([0, 1], individuals)

    for ind in individuals:
        assert math.isfinite(ind.crowd_distance), \
            f"两个个体时拥挤距离应为有限值，实际为 {ind.crowd_distance}"


def test_empty_front():
    """空前沿不应崩溃"""
    nsga = NSGAII()
    nsga.crowding_distance([], [])  # 不应抛出异常


def test_single_individual():
    """单个个体不应崩溃"""
    individuals = [Individual([1], [1.0, 2.0])]

    nsga = NSGAII()
    nsga.crowding_distance([0], individuals)

    assert math.isfinite(individuals[0].crowd_distance), \
        f"单个个体拥挤距离应为有限值，实际为 {individuals[0].crowd_distance}"


if __name__ == "__main__":
    test_identical_fitness_no_inf()
    print("通过: 相同适应度无inf")

    test_identical_fitness_no_nan()
    print("通过: 相同适应度无nan")

    test_boundary_individuals_get_large_distance()
    print("通过: 边界个体获得较大拥挤距离")

    test_two_individuals()
    print("通过: 两个个体正常计算")

    test_empty_front()
    print("通过: 空前沿不崩溃")

    test_single_individual()
    print("通过: 单个个体不崩溃")

    print("\n所有拥挤距离测试通过!")
