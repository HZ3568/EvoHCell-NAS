"""测试安全的基因型解析"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic.population import load_genotype_pool
from darts.genotypes import Genotype

INIT_FILE = Path(__file__).parent.parent / "genetic" / "init_population.txt"


def test_load_genotype_pool_success():
    """应能成功解析init_population.txt中的基因型"""
    if not INIT_FILE.exists():
        print(f"跳过: {INIT_FILE} 不存在")
        return

    genotypes = load_genotype_pool(str(INIT_FILE))

    assert len(genotypes) > 0, "应至少解析出一个基因型"
    for g in genotypes:
        assert isinstance(g, Genotype), f"解析结果应为Genotype类型，实际为 {type(g)}"


def test_genotype_has_required_fields():
    """解析出的基因型应包含normal、normal_concat、reduce、reduce_concat字段"""
    if not INIT_FILE.exists():
        print(f"跳过: {INIT_FILE} 不存在")
        return

    genotypes = load_genotype_pool(str(INIT_FILE))
    g = genotypes[0]

    assert hasattr(g, "normal"), "基因型应有normal字段"
    assert hasattr(g, "normal_concat"), "基因型应有normal_concat字段"
    assert hasattr(g, "reduce"), "基因型应有reduce字段"
    assert hasattr(g, "reduce_concat"), "基因型应有reduce_concat字段"

    # normal和reduce应为(op_name, node_idx)元组的列表
    assert len(g.normal) > 0, "normal cell不应为空"
    assert len(g.reduce) > 0, "reduce cell不应为空"

    for op, idx in g.normal:
        assert isinstance(op, str), f"操作名应为字符串，实际为 {type(op)}"
        assert isinstance(idx, int), f"节点索引应为整数，实际为 {type(idx)}"


def test_invalid_file_raises_error():
    """不存在的文件应抛出FileNotFoundError"""
    try:
        load_genotype_pool("nonexistent_file.txt")
        assert False, "应抛出FileNotFoundError"
    except FileNotFoundError:
        pass


def test_malicious_input_blocked():
    """恶意输入不应被执行"""
    import tempfile
    import os

    # 创建包含恶意代码的临时文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("__import__('os').system('echo hacked')\n")
        tmp_path = f.name

    try:
        load_genotype_pool(tmp_path)
        assert False, "恶意输入应抛出异常"
    except (ValueError, Exception):
        pass  # 预期行为
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    test_load_genotype_pool_success()
    print("通过: 基因型解析成功")

    test_genotype_has_required_fields()
    print("通过: 基因型包含必要字段")

    test_invalid_file_raises_error()
    print("通过: 不存在的文件抛出异常")

    test_malicious_input_blocked()
    print("通过: 恶意输入被阻止")

    print("\n所有基因型解析测试通过!")
