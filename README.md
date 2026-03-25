# 异构DARTS神经架构搜索

基于进化算法的异构DARTS架构搜索，使用零样本评估方法。

## 🎯 核心创新

- **异构架构**: 不同层使用不同的genotype（传统DARTS所有层使用相同genotype）
- **进化搜索**: 使用NSGA-II多目标优化算法
- **零样本评估**: 使用synflow/grad_norm快速评估架构质量
- **多目标优化**: 同时优化架构性能和参数量

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision numpy matplotlib
```

### 2. 运行测试验证修复

```bash
python test_phase1_fixes.py
```

### 3. 运行完整实验

```bash
# 使用默认配置
python real_train.py

# 或自定义参数
python real_train.py --generations 20 --epochs 50 --seed 42
```

### 4. 查看结果

实验结果保存在 `results/exp_TIMESTAMP/` 目录下：
- `summary.json` - 实验总结
- `search_results/pareto_front.json` - Pareto前沿
- `training/best_model.pth` - 最佳模型

## 📖 详细文档

- [实验指南](README_EXPERIMENT.md) - 完整的使用说明和示例
- [第一阶段报告](PHASE1_REPORT.md) - Bug修复详情和改进说明
- [重构计划](C:\Users\21941\.claude\plans\transient-riding-tarjan.md) - 完整的重构路线图

## 🔧 第一阶段修复（已完成）

✅ **变异算子** - 实现真正的变异操作（之前是空操作）
✅ **辅助头训练** - 修复辅助头从未训练的bug
✅ **安全解析** - 替换eval()为安全的genotype解析
✅ **拥挤距离** - 修复NSGA-II中的inf值问题
✅ **实验入口** - 完整的命令行接口和实验管道

## 📊 实验配置

### 统一配置系统

配置参数按5个类别组织：

1. **Architecture** - 网络结构参数（层数、通道数、辅助头）
2. **Evolution** - 进化搜索参数（种群、代数、交叉变异）
3. **Training** - 训练参数（优化器、正则化、数据增强）
4. **Data** - 数据参数（数据集、批大小、加载器）
5. **System** - 系统参数（路径、设备、日志）

详细文档请查看 [CONFIG_GUIDE.md](CONFIG_GUIDE.md)

### 默认配置

```python
# 架构
layers = 8
init_channels = 16
auxiliary = True

# 进化搜索
generations = 20
population_size = 50
pc_layer = 0.5  # 交叉概率
pm_layer = 0.15 # 变异概率

# 训练
epochs = 50
batch_size = 96
learning_rate = 0.025
```

### 使用配置文件

创建 `experiments/my_config.json`:

```json
{
  "evolution": {
    "generations": 30,
    "population_size": 100,
    "seed": 42
  },
  "training": {
    "epochs": 100
  }
}
```

运行:
```bash
python real_train.py --config experiments/my_config.json
```

### 使用预设

```bash
# 快速测试（2代，5轮）
python real_train.py --preset quick_test

# 标准实验（20代，100轮）
python real_train.py --preset standard

# 大规模实验（50代，600轮）
python real_train.py --preset large
```

## 🎓 项目结构

```
zero_train_for_darts/
├── darts/                      # DARTS架构定义
│   ├── genotypes.py           # Genotype定义
│   ├── model.py               # 同构DARTS模型
│   ├── model_hetero_cell.py   # 异构DARTS模型（核心创新）
│   └── operations.py          # 操作原语
├── genetic/                    # 进化算法
│   ├── NSGA_II.py             # NSGA-II实现
│   ├── population.py          # 种群管理
│   ├── evaluate.py            # 零样本评估
│   ├── crossover_and_mutation.py  # 交叉和变异
│   └── config.py              # 配置
├── zero_cost/                  # 零样本指标
│   ├── synflow.py             # SynFlow指标
│   └── grad_norm.py           # Grad-Norm指标
├── real_train.py              # 主入口（实验启动）
├── test_phase1_fixes.py       # 测试脚本
└── experiments/               # 实验配置
    └── example_config.json
```

## 💡 使用示例

### 示例1: 基线实验
```bash
python real_train.py \
  --generations 20 \
  --seed 42 \
  --save_dir ./results/baseline
```

### 示例2: 高变异率实验
```bash
python real_train.py \
  --pm_layer 0.3 \
  --seed 42 \
  --save_dir ./results/high_mutation
```

### 示例3: 仅运行搜索
```bash
python real_train.py \
  --mode search \
  --generations 20 \
  --save_dir ./results/search_only
```

## 📈 预期改进

基于第一阶段的bug修复，预期：

- **Pareto前沿质量**: 提升10-30%（变异算子修复）
- **最终准确率**: 提升0.5-1.5%（辅助头修复）
- **数值稳定性**: 更稳定的NSGA-II选择（拥挤距离修复）

## 🔜 下一步计划

### 第二阶段：验证框架（进行中）
- [ ] 完全可复现性（种子设置）
- [ ] 正确的训练/验证/测试集划分
- [ ] 多batch零样本评估
- [ ] 实验跟踪系统

### 第三阶段：基线对比
- [ ] 同构DARTS基线
- [ ] 随机搜索基线
- [ ] 超参数消融实验

### 第四阶段：代码质量
- [ ] 消除代码重复
- [ ] 清理注释代码
- [ ] 添加类型提示和文档

## 🤝 贡献

这是一个研究项目，欢迎提出改进建议！

## 📝 引用

如果使用本代码，请引用：
- DARTS: Differentiable Architecture Search
- NSGA-II: A Fast and Elitist Multiobjective Genetic Algorithm
- Zero-Cost Proxies for Neural Architecture Search

## ⚠️ 注意事项

1. 第一次运行会自动下载CIFAR-10数据集
2. 建议使用GPU进行训练（CPU会很慢）
3. 完整实验（20代+50轮训练）需要数小时
4. 可以先用小参数测试：`--generations 2 --epochs 1`

## 📧 联系

如有问题，请查看文档或提issue。
