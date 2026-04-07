from bananas.nas_algorithms import bananas
from bananas.data import Data

train_arch = []

for i in range(10):
    data = bananas(Data('darts'), train_arch, num_init=10, k=5, total_queries=20)
    train_arch.append(data)

print(train_arch)
