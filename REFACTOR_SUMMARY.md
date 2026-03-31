# EvoHCell-NAS 架构保存/读取逻辑重构说明

## 重构概述

本次重构优化了 search.py 和 train.py 之间的架构传递方式，使其更加清晰、明确，避免隐式行为。

## 主要变化

### 1. search.py 的变化

#### 1.1 移除冗余字段
- **变更前**: `top_candidates.json` 中每个 genotype 包含 `normal_str` 和 `reduce_str` 字段
- **变更后**: 只保留结构化字段 (`normal`, `normal_concat`, `reduce`, `reduce_concat`)
- **原因**: 字符串字段仅用于人工查看，不应混入机器可读的 JSON 中

#### 1.2 新增可读文本文件
- **新文件**: `top_candidates_readable.txt`
- **用途**: 专门用于人工查看架构的字符串表示
- **格式**: 包含搜索配置、每个候选的详细信息和层级架构

#### 1.3 单独导出每个候选架构
- **新文件**: `candidate_0.json`, `candidate_1.json`, ..., `candidate_{k-1}.json`
- **用途**: 每个文件包含单个候选的完整信息，可直接作为 train.py 的输入
- **内容**: 包含 `genotype_list`、`zero_cost_score`、`params_mb` 等关键信息

#### 1.4 输出文件总览
搜索完成后，`results/search_xxx/` 目录包含：
```
results/search_xxx/
├── top_candidates.json          # 所有候选的汇总（机器可读）
├── top_candidates_readable.txt  # 所有候选的可读格式（人工查看）
├── candidate_0.json             # 第 0 个候选（Pareto 前沿最优）
├── candidate_1.json             # 第 1 个候选
├── ...
├── candidate_4.json             # 第 k-1 个候选
├── pareto_front.png             # 帕累托前沿可视化
└── search.log                   # 搜索日志
```

### 2. train.py 的变化

#### 2.1 删除隐式逻辑
- **变更前**: 支持传入 `top_candidates.json`，会默认取第一个候选
- **变更后**: 不再支持 `top_candidates.json`，必须明确指定单个候选文件
- **原因**: 隐式行为容易造成混淆，用户应明确知道训练的是哪个架构

#### 2.2 简化输入方式
现在只支持两种明确的输入方式：

**方式 1: 使用预定义架构**
```bash
python train.py --arch DARTS
```

**方式 2: 使用搜索得到的单个候选**
```bash
python train.py --genotype_json ./results/search_xxx/candidate_0.json
```

#### 2.3 错误提示优化
如果用户错误地传入 `top_candidates.json`，会收到明确的错误提示：
```
genotype_json 格式不正确：必须是 list，或包含 key 'genotype_list'。
提示：请使用 search.py 生成的 candidate_*.json 文件，而不是 top_candidates.json。
```

## 使用示例

### 完整工作流

**步骤 1: 运行搜索**
```bash
python search.py --generations 30 --population_size 50 --top_k 5
```

**步骤 2: 查看候选架构**
```bash
# 查看可读格式
cat ./results/search_xxx/top_candidates_readable.txt

# 或查看 JSON 格式
cat ./results/search_xxx/candidate_0.json
```

**步骤 3: 训练选定的架构**
```bash
# 训练第 0 个候选（通常是 Pareto 前沿最优）
python train.py --genotype_json ./results/search_xxx/candidate_0.json --epochs 600

# 或训练第 1 个候选
python train.py --genotype_json ./results/search_xxx/candidate_1.json --epochs 600
```

## 关键代码变更

### search.py 关键修改

1. **移除字符串字段附加函数**
```python
# 删除了 _attach_readable_genotype_fields()
# 新增了 _genotype_to_readable_str() 用于生成可读文本
```

2. **新增可读文本生成**
```python
# 生成 top_candidates_readable.txt
readable_path = Path(save_dir) / "top_candidates_readable.txt"
with open(readable_path, "w", encoding="utf-8") as f:
    # 写入搜索配置和每个候选的详细信息
```

3. **新增单独候选文件导出**
```python
# 为每个候选单独导出 JSON 文件
for candidate in candidates:
    candidate_file = Path(save_dir) / f"candidate_{candidate['id']}.json"
    candidate_data = {
        "id": candidate["id"],
        "genotype_list": candidate["genotype_list"],
        "zero_cost_score": candidate["zero_cost_score"],
        "params_mb": candidate["params_mb"],
        # ...
    }
```

### train.py 关键修改

1. **简化 load_genotype_list() 函数**
```python
def load_genotype_list():
    """
    支持两种输入方式：
    1. --arch <name>: 从 darts/genotypes.py 加载预定义架构
    2. --genotype_json <path>: 从单个候选 JSON 文件加载
    """
    # 删除了从 top_candidates.json 取第一个候选的逻辑
```

2. **更新参数说明**
```python
parser.add_argument('--genotype_json', type=str, default=None,
                    help='path to a single candidate json file (e.g., candidate_0.json)')
```

## 兼容性说明

### JSON 格式兼容性
- ✅ genotype 的 JSON 结构保持不变（`normal`, `normal_concat`, `reduce`, `reduce_concat`）
- ✅ 可以正常恢复为 `Genotype` 对象
- ✅ 现有的 `dict_to_genotype()` 函数无需修改

### 向后兼容性
- ❌ 不再支持直接使用 `top_candidates.json` 作为 train.py 的输入
- ✅ 仍然支持 `--arch` 方式加载预定义架构
- ✅ 仍然支持直接传入 genotype list 的 JSON 文件

## 优势总结

1. **清晰性**: 用户明确知道训练的是哪个候选架构
2. **可维护性**: 代码逻辑更简单，没有隐式行为
3. **易用性**: 单个候选文件可以直接传递和分享
4. **可读性**: 提供专门的文本文件供人工查看
5. **灵活性**: 可以轻松训练任意候选，而不仅仅是第一个

## 现在训练的是什么？

使用 `train.py` 时，训练的架构取决于输入参数：

- **使用 `--arch DARTS`**: 训练 `darts/genotypes.py` 中定义的 DARTS 架构
- **使用 `--genotype_json candidate_0.json`**: 训练 `candidate_0.json` 文件中的架构
  - 该文件由 `search.py` 生成
  - 包含 20 层（或 `--layers` 指定的层数）的异构 cell 架构
  - 每层的 normal cell 和 reduce cell 结构都在 `genotype_list` 中定义
  - 该候选在搜索过程中的 Pareto 前沿排名为 `front_rank`

**示例**：
```bash
python train.py --genotype_json ./results/search_20250330_123456/candidate_0.json
```
这将训练：
- 文件：`./results/search_20250330_123456/candidate_0.json`
- 架构：该文件中 `genotype_list` 定义的 20 层异构架构
- 来源：搜索过程中发现的第 0 个候选（通常是 Pareto 前沿最优解）
