from bananas.nas_algorithms import bananas
from bananas.data import Data
from utils.convert_to_genotype import convert_to_genotype

train_arch = []

# 5 * 20 = 100
for i in range(5):
    data = bananas(Data('darts'), train_arch, num_init=10, k=5, total_queries=20)
    train_arch.append(data)

print(train_arch)

data = sorted([(d['val_loss'], d['spec']) for d in train_arch], key=lambda d: d[0])[:30]

# 将data存入txt文件中
for d in data:
    arch = convert_to_genotype(d[1])
    with open('../results/arch_pool/arch.txt', 'a') as f:
        f.write(str(arch) + '\n')
