import time
import numpy as np
import torch
from bananas.acquisition_functions import acq_fn
from bananas.meta_neural_net import MetaNeuralnet
from utils.logger_utils import setup_logger

logger = setup_logger(name="arch_pool", save_dir="../results/arch_pool", level="INFO")


def bananas(search_space,
            data,
            num_init=0,
            k=2,
            loss='val_loss',
            total_queries=150,
            num_ensemble=5,
            acq_opt_type='mutation',
            num_arches_to_mutate=1,
            explore_type='its',
            encoding_type='trunc_path',
            cutoff=40,
            deterministic=True,
            verbose=1, ):
    # # 读取提前预处理的架构数据集
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # file_path = os.path.join(current_dir, 'trained_arch.txt')
    # with open(file_path, 'r') as file:
    #     lines = file.readlines()
    #
    # for i in range(0, len(lines), 3):
    #     d = dict()
    #     L1 = lines[i].strip()
    #     arch_str = L1[len("arch: "):]
    #     arch = ast.literal_eval(arch_str)
    #
    #     arc = Arch(arch)
    #     encoding = arc.encode_paths()[:40]
    #
    #     L2 = lines[i + 1].strip()
    #     val_loss_str = L2.split(":")[1]
    #     val_loss = float(val_loss_str)
    #
    #     d['spec'] = arch
    #     d['encoding'] = encoding
    #     d['val_loss'] = val_loss
    #
    #     data.append(d)

    if len(data) == 0:
        data = search_space.generate_random_dataset(num=num_init,
                                                    encoding_type=encoding_type,
                                                    cutoff=cutoff,
                                                    deterministic_loss=deterministic)
        query = len(data) + k
    else:
        query = k

    new_data = []

    while query <= total_queries:
        logger.info("query: %d" % query)

        xtrain = np.array([d['encoding'] for d in data])  # 路径编码之后的架构
        ytrain = np.array([d[loss] for d in data])  # valid loss

        if (query == num_init + k) and verbose:
            print('bananas xtrain shape', xtrain.shape)
            print('bananas ytrain shape', ytrain.shape)

        # 通过对原架构变异的方式，得到一组候选架构，该候选架构没有被训练过
        candidates = search_space.get_candidates(data,
                                                 acq_opt_type=acq_opt_type,
                                                 encoding_type=encoding_type,
                                                 cutoff=cutoff,
                                                 num_arches_to_mutate=num_arches_to_mutate,
                                                 loss=loss)

        xcandidates = np.array([c['encoding'] for c in candidates])
        candidate_predictions = []

        # 训练一组神经网络
        start_time = time.time()
        train_error = 0
        for _ in range(num_ensemble):
            meta_neuralnet = MetaNeuralnet()
            train_error += meta_neuralnet.fit(xtrain, ytrain)  # xtrain的长度在逐渐增加

            # 预测候选架构的valid loss
            candidate_predictions.append(np.squeeze(meta_neuralnet.predict(xcandidates)))

            del meta_neuralnet
            torch.cuda.empty_cache()

        train_error /= num_ensemble
        logger.info("query = %d, Meta neural net train error = %f" % (query, train_error))
        logger.info("prediction time %s" % (time.time() - start_time))

        # 为所有候选解计算采集函数
        candidate_indices = acq_fn(candidate_predictions, explore_type)

        # 根据采集函数值，选出最具潜力的前 k 个架构进行真实评估
        for i in candidate_indices[:k]:
            arch_dict = search_space.query_arch(candidates[i]['spec'],
                                                encoding_type=encoding_type,
                                                cutoff=cutoff)
            new_data.append(arch_dict)

        query += k

    return new_data

# 已搞明白
def random_search(search_space,
                  total_queries=150,
                  loss='val_loss',  # 这里并没有违规，使用的是valid的err，没有使用test
                  deterministic=True,
                  verbose=1):  # search_space是一个Data对象
    """
    random search
    """
    data = search_space.generate_random_dataset(num=total_queries,
                                                encoding_type='adj',
                                                deterministic_loss=deterministic)

    if verbose:
        top_5_loss = sorted([d[loss] for d in data])[:min(5, len(data))]
        print('random, query {}, top 5 losses {}'.format(total_queries, top_5_loss))
    return data


# 已搞明白
def evolution_search(search_space,
                     total_queries=150,
                     num_init=10,
                     k=10,
                     loss='val_loss',
                     population_size=30,
                     tournament_size=10,
                     mutation_rate=1.0,
                     deterministic=True,
                     regularize=True,
                     verbose=1):
    """
    regularized evolution
    """
    data = search_space.generate_random_dataset(num=num_init, deterministic_loss=deterministic)

    losses = [d[loss] for d in data]
    query = num_init
    population = [i for i in range(min(num_init, population_size))]  # 下标

    while query <= total_queries:

        # 通过从种群的 随机子集 中变异出最佳架构来进化种群
        sample = np.random.choice(population, tournament_size)
        # 第一个0代表loss最小的架构，第二个0代表元组的第一个位置：下标
        best_index = sorted([(i, losses[i]) for i in sample], key=lambda i: i[1])[0][0]
        mutated = search_space.mutate_arch(data[best_index]['spec'],
                                           mutation_rate=mutation_rate)
        arch_dict = search_space.query_arch(mutated)
        data.append(arch_dict)
        losses.append(arch_dict[loss])
        population.append(len(data) - 1)

        # population的意义在于进化, data库中的所有架构最后会统一评选。
        """
            问题：在种群规模较小时，使用 regularize=True（淘汰最老个体）好，还是 regularize=False（淘汰最差个体）更好？
            gpt解释：当种群较小时，建议使用 regularize=True（淘汰最老个体），这样可以保持 多样性（diversity），避免算法陷入局部最优。
            而 regularize=False 更偏向贪婪策略，适合在大种群或收敛阶段使用，可能更快收敛但容易过早丧失探索能力。
        """
        if len(population) >= population_size:
            if regularize:
                oldest_index = sorted([i for i in population])[0]  # 下标越小，越老
                population.remove(oldest_index)
            else:
                worst_index = sorted([(i, losses[i]) for i in population], key=lambda i: i[1])[-1][0]
                population.remove(worst_index)  # 这里只需要维护列表population, data和losses不需要remove

        if verbose and (query % k == 0):
            top_5_loss = sorted([d[loss] for d in data])[:min(5, len(data))]
            print('evolution, query {}, top 5 losses {}'.format(query, top_5_loss))

        query += 1

    return data


"""
    核心思想
    通过训练Meta Neural Network来预测不同网络架构的性能，然后用贝叶斯优化方法选出最可能优异的结构进行评估，从而高效探索架构空间。
"""





if __name__ == '__main__':
    # datas = []
    # with open('trained_arch.txt', 'r') as file:
    #     lines = file.readlines()
    #
    # for i in range(0, len(lines), 3):
    #     data = dict()
    #     L1 = lines[i].strip()
    #     arch_str = L1[len("arch: "):]
    #     arch = ast.literal_eval(arch_str)
    #
    #     arc = Arch(arch)
    #     encoding = arc.encode_paths()[:40]
    #
    #     L2 = lines[i + 1].strip()
    #     val_loss_str = L2.split(":")[1]
    #     val_loss = float(val_loss_str)
    #
    #     data['spec'] = arch
    #     data['encoding'] = encoding
    #     data['val_loss'] = val_loss
    #
    #     print(data['spec'], data['encoding'], data['val_loss'])
    #
    #     datas.append(data)
    #
    # xtrain = np.array([data['encoding'] for data in datas])
    # ytrain = np.array([data['val_loss'] for data in datas])
    #
    # start = time.time()
    # meta_neuralnet = MetaNeuralnet()
    # train_loss = meta_neuralnet.fit(xtrain, ytrain)  # xtrain的长度在逐渐增加
    #
    # print('training time: {}'.format(time.time() - start))
    #
    # d = Data('darts')
    #
    # candidates = d.get_candidates(datas,
    #                               acq_opt_type='mutation',
    #                               encoding_type='trunc_path',
    #                               cutoff=40,
    #                               num_arches_to_mutate=1,
    #                               loss='val_loss')
    #
    # xcandidates = np.array([c['encoding'] for c in candidates])
    #
    # start = time.time()
    # # 预测候选架构的valid loss
    # print(meta_neuralnet.predict(xcandidates))
    # print('prediction time: {}'.format(time.time() - start))
    pass
