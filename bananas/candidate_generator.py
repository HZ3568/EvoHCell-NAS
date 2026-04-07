from bananas.nas_algorithms import bananas
from bananas.data import Data
from utils.convert_to_genotype import convert_to_genotype
from pathlib import Path


def main():
    train_arch = []

    # 5 * 20 = 100
    for i in range(5):
        data = bananas(Data('darts'), train_arch, num_init=10, k=5, total_queries=20)
        train_arch.append(data)

    print(train_arch)

    top_data = sorted(
        [(d['val_loss'], d['spec']) for d in train_arch],
        key=lambda d: d[0]
    )[:30]

    output_path = Path("../results/arch_pool/arch.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        for d in top_data:
            arch = str(convert_to_genotype(d[1]))
            f.write(f"{arch} valid_loss:{d[0]:.4f}\n")


if __name__ == "__main__":
    main()