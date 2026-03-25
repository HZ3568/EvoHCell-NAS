"""配置加载工具

从experiments/config.json加载实验配置
"""
import json
from pathlib import Path
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """加载默认配置文件

    Returns:
        配置字典
    """
    config_path = Path(__file__).parent.parent / "experiments" / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def get_evolution_config() -> Dict[str, Any]:
    """获取进化搜索配置"""
    config = load_config()
    evo_config = config.get("进化搜索", {}).copy()
    evo_config["layers"] = config.get("架构", {}).get("layers", 8)
    evo_config["objectives"] = 2  # 固定为2目标
    return evo_config


def get_train_config() -> Dict[str, Any]:
    """获取训练配置"""
    config = load_config()
    train_config = {}
    train_config.update(config.get("架构", {}))
    train_config.update(config.get("训练", {}))
    train_config.update(config.get("数据", {}))
    train_config["save_dir"] = config.get("系统", {}).get("save_dir", "./results") + "/final_train"
    return train_config
