# 异构DARTS神经架构搜索

基于进化算法的异构DARTS架构搜索，使用零样本评估方法。

## 🎯 核心创新

- **异构架构**: 不同层使用不同的genotype（传统DARTS所有层使用相同genotype）
- **进化搜索**: 使用NSGA-II多目标优化算法
- **零样本评估**: 使用synflow/grad_norm快速评估架构质量
- **多目标优化**: 同时优化架构性能和参数量


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


## 🔜 TODO
- [ ] pipline.py
- [ ] 修改train的输出日志
- [ ] 检查search的逻辑
- [ ] 设计交叉和变异算法
- [ ] 零样本评估指标 取负改成取倒数（尝试）
