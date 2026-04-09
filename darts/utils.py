import os
import time
import torch
import numpy as np
import torchvision.transforms as transforms
import logging
from collections import namedtuple
from pathlib import Path


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def data_transforms_cifar10(cutout, cutout_length):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if cutout:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.bernoulli(torch.full((x.size(0), 1, 1, 1), keep_prob, device=x.device, dtype=x.dtype))
        x = x.div(keep_prob) * mask
    return x


def create_exp_dir(stage) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).resolve().parent.parent  # 项目根目录
    path = project_root / "results" / f"{stage}_{timestamp}"

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    return str(path)


def setup_logger(name: str, save_dir: str | None = None, level: str = "INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(save_dir) / f"{name}.log", mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def convert_list_to_genotype(arch):
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
