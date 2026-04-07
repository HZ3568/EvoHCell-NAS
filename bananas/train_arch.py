import argparse
import time
import logging
import sys
import os
import ast

from bananas.arch import Arch
from bananas.train_class import Train


def load_architecture_from_txt(filepath):
    archs = []
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # 跳过空行
                arch = ast.literal_eval(line)
                archs.append(arch)
    return archs


def save_architecture_to_txt(arch, filepath, val_accs):
    with open(filepath, 'a') as file:
        file.write(f"arch: {str(arch)}\n")

        # Average validation loss
        val_sum = sum(100 - val_acc for val_acc in val_accs)
        val_loss_avg = val_sum / len(val_accs)
        file.write(f"val_loss: {val_loss_avg}\n")
        file.write("\n")


def run(args):
    untrained_filepath = os.path.expanduser(args.untrained_filepath)
    trained_filepath = os.path.expanduser(args.trained_filepath)
    epochs = args.epochs
    gpu = args.gpu
    train_portion = args.train_portion
    seed = args.seed
    save = args.save

    archs = []
    for i in range(50):
        arch = Arch.random_arch()
        archs.append(arch)

    # 在 untrained_arch.txt 中加载 arch
    # archs = load_architecture_from_txt(untrained_filepath)
    print('loaded archs', archs)

    for arch in archs:
        # train an arch
        trainer = Train()
        val_accs = trainer.main(arch,
                                epochs=epochs,
                                gpu=gpu,
                                train_portion=train_portion,
                                seed=seed,
                                save=save)

        # Save the architecture and loss information to a txt file
        save_architecture_to_txt(arch, trained_filepath, val_accs)

    # Output the final results
    print(f"Training complete. Results saved to {trained_filepath}")


def main(args):
    # set up save dir
    save_dir = '../'

    # set up logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for training a darts arch')
    parser.add_argument('--untrained_filepath', type=str, default='untrained_arch.txt',
                        help='name of input text file containing architecture')
    parser.add_argument('--trained_filepath', type=str, default='trained_arch.txt',
                        help='name of output text file for architecture and loss')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')  # 50
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    parser.add_argument('--train_portion', type=float, default=0.7, help='portion of training data used for training')
    parser.add_argument('--seed', type=float, default=0, help='random seed to use')
    parser.add_argument('--save', type=str, default='EXP', help='directory to save to')

    args = parser.parse_args()
    main(args)
