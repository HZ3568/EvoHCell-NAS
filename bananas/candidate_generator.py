from bananas.nas_algorithms import bananas
from bananas.data import Data
from pathlib import Path
from darts.utils import create_exp_dir, setup_logger, convert_list_to_genotype



def main():
    save_dir = create_exp_dir(stage="arch_pool")
    setup_logger(name="arch_pool", save_dir=save_dir, level="INFO")
    arch_path = Path(create_exp_dir(stage="arch_pool")) / "arch.txt"

    train_arch = []

    # 5 * 20 = 100
    for i in range(5):
        data = bananas(Data('darts'), train_arch, num_init=10, k=5, total_queries=20)
        train_arch.extend(data)

    top_data = sorted(
        [(d['val_loss'], d['spec']) for d in train_arch],
        key=lambda d: d[0]
    )[:30]

    with open(arch_path, "a", encoding="utf-8") as f:
        for d in top_data:
            arch = str(convert_list_to_genotype(d[1]))
            f.write(f"{arch} valid_loss:{d[0]:.4f}\n")


if __name__ == "__main__":
    main()