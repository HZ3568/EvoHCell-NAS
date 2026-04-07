from collections import namedtuple


def convert_to_genotype(arch):
    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
    op_dict = {
        0: 'none',
        1: 'max_pool_3x3',
        2: 'avg_pool_3x3',
        3: 'skip_connect',
        4: 'sep_conv_3x3',
        5: 'sep_conv_5x5',
        6: 'dil_conv_3x3',
        7: 'dil_conv_5x5'
    }

    darts_arch = [[], []]
    i = 0
    for cell in arch:
        for n in cell:
            darts_arch[i].append((op_dict[n[1]], int(n[0])))
        i += 1
    geno = Genotype(normal=darts_arch[0], normal_concat=[2, 3, 4, 5], reduce=darts_arch[1],
                    reduce_concat=[2, 3, 4, 5])
    return str(geno)


if __name__ == '__main__':
    arch = ([(0, 0), (1, 6), (2, 6), (0, 7), (1, 6), (3, 2), (2, 4), (0, 2)],
            [(0, 0), (1, 7), (1, 7), (0, 5), (3, 2), (1, 0), (3, 4), (2, 2)])

    genotype = convert_to_genotype(arch)
    print(genotype)
