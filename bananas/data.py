import sys
import os
import logging
from bananas.arch import Arch


class Data:

    def __init__(self,
                 search_space,
                 dataset='cifar100',
                 nasbench_folder='./',
                 loaded_nasbench=None):
        self.search_space = search_space
        self.dataset = dataset

        if search_space != 'darts':
            print(search_space, 'is not a valid search space')
            sys.exit()

    def get_type(self):
        return self.search_space

    @staticmethod
    def query_arch(arch=None,
                   train=True,
                   encoding_type='path',
                   cutoff=-1,
                   epochs=0):

        arch_dict = dict()
        arch_dict['epochs'] = epochs

        if arch is None:
            arch = Arch.random_arch()

        arch_dict['spec'] = arch

        if encoding_type == 'path':  # 完整的路径编码
            encoding = Arch(arch).encode_paths()
        elif encoding_type == 'trunc_path':  # 删减后的路径编码
            encoding = Arch(arch).encode_paths()[:cutoff]
        else:  # 无编码
            encoding = arch

        arch_dict['encoding'] = encoding

        if train:
            if epochs == 0:
                epochs = 10  # 50
            arch_dict['val_loss'] = Arch(arch).query(epochs=epochs)

        return arch_dict

    def mutate_arch(self, arch, mutation_rate=1.0):
        if self.search_space == 'darts':
            return Arch(arch).mutate(int(mutation_rate))

    def get_hash(self, arch):
        # return the path indices of the architecture, used as a hash
        if self.search_space == 'darts':
            return Arch(arch).get_path_indices()[0]

    def generate_random_dataset(self,
                                num=10,
                                train=True,
                                encoding_type='path',
                                cutoff=-1,
                                allow_isomorphisms=False,
                                deterministic_loss=True,
                                patience_factor=5):
        """
        创建随机采样架构的数据集
        使用路径索引的哈希映射测试同构性
        使用 patience_factor 避免无限循环
        """
        data = []
        dic = {}
        tries_left = num * patience_factor
        while len(data) < num:  # num是架构的数量
            tries_left -= 1
            if tries_left <= 0:
                break
            logging.info('arch idx = %d', len(data))
            arch_dict = self.query_arch(train=train,
                                        encoding_type=encoding_type,
                                        cutoff=cutoff)

            h = self.get_hash(arch_dict['spec'])
            if allow_isomorphisms or h not in dic:
                dic[h] = 1
                data.append(arch_dict)

        return data  # data的结构 : [{},{},...,{}], {}表示一个架构的所有信息

    def get_candidates(self,
                       data,
                       train_idxs,
                       valid_idxs,
                       num=100,  # 变异的架构需要多一些，这样才能增加数据的多样性，这些数据不需要经过训练。
                       acq_opt_type='mutation',
                       encoding_type='path',
                       cutoff=-1,
                       loss='val_loss',
                       patience_factor=5,
                       num_arches_to_mutate=1,
                       max_mutation_rate=3,
                       allow_isomorphisms=False):
        """
        Creates a set of candidate architectures with mutation or random or mutation_random
        """

        candidates = []
        # set up hash map
        dic = {}
        for d in data:
            arch = d['spec']
            h = self.get_hash(arch)
            dic[h] = 1

        if acq_opt_type in ['mutation']:
            top_k = 10  # 前提是预处理10个架构
            # 通过从低到高排序损失值，选择需要被变异的父代
            best_arches = [arch['spec'] for arch in sorted(data, key=lambda i: i[loss])[:top_k]]

            for arch in best_arches:
                if len(candidates) >= num:  # 候选架构的数量为sum
                    break
                for i in range(num // top_k // max_mutation_rate):
                    for rate in range(1, max_mutation_rate + 1):
                        mutated = self.mutate_arch(arch, mutation_rate=rate)
                        arch_dict = self.query_arch(train_idxs,
                                                    valid_idxs,
                                                    mutated,
                                                    train=False,  # 只采集架构，而不训练
                                                    encoding_type=encoding_type,
                                                    cutoff=cutoff)
                        h = self.get_hash(mutated)

                        if allow_isomorphisms or h not in dic:
                            dic[h] = 1
                            candidates.append(arch_dict)

        return candidates

    def remove_duplicates(self, candidates, data):
        # input: two sets of architectues: candidates and data
        # output: candidates with arches from data removed

        dic = {}
        for d in data:
            dic[self.get_hash(d['spec'])] = 1
        unduplicated = []
        for candidate in candidates:
            if self.get_hash(candidate['spec']) not in dic:
                dic[self.get_hash(candidate['spec'])] = 1
                unduplicated.append(candidate)
        return unduplicated

    def encode_data(self, dicts):
        """
        method used by metann_runner.py (for Arch)
        input: list of arch dictionary objects
        output: xtrain (encoded architectures), ytrain (val loss)
        """
        data = []

        for dic in dicts:
            arch = dic['spec']
            encoding = Arch(arch).encode_paths()
            data.append((arch, encoding, dic['val_loss_avg'], None))

        return data
