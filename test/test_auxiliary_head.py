"""测试辅助头训练正常工作"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from darts.model_hetero_cell import NetworkCIFARHeteroCell
from darts.genotypes import DARTS_V2


def _make_model(auxiliary: bool):
    """创建测试用模型"""
    genotype_list = [DARTS_V2] * 8
    return NetworkCIFARHeteroCell(
        C=16, num_classes=10, layers=8,
        auxiliary=auxiliary, genotype_list=genotype_list,
    )


def test_training_mode_returns_tuple():
    """训练模式下带辅助头的模型应返回 (logits, logits_aux) 元组"""
    model = _make_model(auxiliary=True)
    model.train()
    x = torch.randn(2, 3, 32, 32)
    output = model(x)

    assert isinstance(output, tuple), f"训练模式应返回元组，实际返回 {type(output)}"
    assert len(output) == 2, f"元组长度应为2，实际为 {len(output)}"

    logits, logits_aux = output
    assert logits.shape == (2, 10), f"logits形状应为(2,10)，实际为 {logits.shape}"
    assert logits_aux.shape == (2, 10), f"logits_aux形状应为(2,10)，实际为 {logits_aux.shape}"


def test_eval_mode_returns_tensor():
    """评估模式下模型应返回单个tensor而非元组"""
    model = _make_model(auxiliary=True)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    output = model(x)

    assert isinstance(output, torch.Tensor), f"评估模式应返回Tensor，实际返回 {type(output)}"
    assert output.shape == (2, 10)


def test_auxiliary_head_receives_gradients():
    """辅助头应能接收梯度"""
    model = _make_model(auxiliary=True)
    model.train()
    x = torch.randn(2, 3, 32, 32)
    logits, logits_aux = model(x)

    loss = logits.sum() + logits_aux.sum()
    loss.backward()

    aux_has_grad = any(p.grad is not None for p in model.auxiliary_head.parameters())
    assert aux_has_grad, "辅助头应接收到梯度"


def test_no_auxiliary_returns_tensor():
    """不使用辅助头时，训练模式也应返回单个tensor"""
    model = _make_model(auxiliary=False)
    model.train()
    x = torch.randn(2, 3, 32, 32)
    output = model(x)

    assert isinstance(output, torch.Tensor), f"无辅助头时应返回Tensor，实际返回 {type(output)}"


if __name__ == "__main__":
    test_training_mode_returns_tuple()
    print("通过: 训练模式返回元组")

    test_eval_mode_returns_tensor()
    print("通过: 评估模式返回tensor")

    test_auxiliary_head_receives_gradients()
    print("通过: 辅助头接收梯度")

    test_no_auxiliary_returns_tensor()
    print("通过: 无辅助头返回tensor")

    print("\n所有辅助头测试通过!")
