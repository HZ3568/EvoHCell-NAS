import random
import torch
import logging
import gc
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
from collections import namedtuple

from darts.model import NetworkCIFAR
from darts import utils
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
from darts.utils import data_transforms_cifar10, convert_list_to_genotype

logger = logging.getLogger("arch_pool")

train_transform, valid_transform = data_transforms_cifar10(False, 16)
train_data = dset.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
valid_data = dset.CIFAR10(root='../data', train=False, download=True, transform=valid_transform)

train_queue = torch.utils.data.DataLoader(
    train_data,
    batch_size=96,
    shuffle=True,
    pin_memory=True,
    num_workers=0
)

valid_queue = torch.utils.data.DataLoader(
    valid_data,
    batch_size=96,
    shuffle=False,
    pin_memory=True,
    num_workers=0
)


class Train:

    def __init__(self):
        self.learning_rate = 0.025
        self.momentum = 0.9
        self.weight_decay = 3e-4
        self.load_weights = 0
        self.report_freq = 50
        self.gpu = 0
        self.epochs = 600
        self.init_channels = 16  # 36
        self.layers = 8  # 20
        self.model_path = 'saved_models'
        self.auxiliary = False  # True
        self.auxiliary_weight = 0.4
        self.cutout = True
        self.cutout_length = 16
        self.drop_path_prob = 0.2
        self.save = 'EXP'
        self.seed = 0
        self.arch = 'BANANAS'
        self.grad_clip = 5
        self.train_portion = 0.7
        self.validation_set = True
        self.CIFAR_CLASSES = 10  # cifar10 = 10, cifar100 = 100

    def main(self, arch, epochs=600, gpu=0, load_weights=False, train_portion=0.7, seed=0):

        # ---------------- 参数设定 ----------------
        self.arch = arch
        self.epochs = epochs
        self.load_weights = load_weights
        self.gpu = gpu
        self.train_portion = train_portion  # train : valid = 7 : 3
        self.seed = seed
        self.validation_set = (self.train_portion < 1)

        print(f'Train and Valid Arch: {arch}')

        # ---------------- 设置设备 ----------------
        device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            logging.info('no gpu device available')
            torch.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            cudnn.benchmark = False
            cudnn.enabled = True
            cudnn.deterministic = True
            logging.info('gpu device = %d' % gpu)

        # ---------------- 模型定义 ----------------
        Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
        genotype = eval(convert_list_to_genotype(arch))
        model = NetworkCIFAR(self.init_channels, self.CIFAR_CLASSES, self.layers, self.auxiliary, genotype)  # 子网模型
        model = model.to(device)

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(
            model.parameters(), self.learning_rate,
            momentum=self.momentum, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs))

        logger.info("train local batch number: %d" % len(train_queue))
        logger.info("valid local batch number: %d" % len(valid_queue))

        # ---------------- 正式训练 ----------------
        valid_accs = []

        for epoch in range(epochs):  # epochs = 10
            model.drop_path_prob = self.drop_path_prob * epoch / epochs

            train_acc, train_obj = self.train(train_queue, model, criterion, optimizer)
            logger.info('epoch = %d, train_acc = %.2f, train_loss = %.2f', epoch, train_acc, train_obj)

            valid_acc, valid_obj = self.infer(valid_queue, model, criterion)
            logger.info('epoch = %d, valid_acc = %.2f, valid_loss = %.2f', epoch, valid_acc, valid_obj)

            scheduler.step()  # 放在 optimizer.step() 之后

            if epoch in list(range(max(0, epochs - 5), epochs)):  # range(n-5, n), 对后5个val_acc取平均，以此为评判标准
                valid_accs.append(valid_acc)

        val_sum = sum(100 - val_acc for val_acc in valid_accs)
        val_loss_avg = val_sum / len(valid_accs)
        logger.info('average valid loss: %f', val_loss_avg)

        del model
        torch.cuda.empty_cache()
        gc.collect()
        return valid_accs

    """
        valid的作用：用valid验证train的效果。这个需要保留。
        test的作用：取出当前查询次数下最小的val_loss, 然后查看它的test_loss。
    """

    def train(self, train_queue, model, criterion, optimizer):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.train()

        device = torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() else 'cpu')

        for step, (input, target) in enumerate(train_queue):
            input = input.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits = model(input)
            loss = criterion(logits, target)

            # if self.auxiliary:
            #     loss_aux = criterion(logits_aux, target)
            #     loss += self.auxiliary_weight * loss_aux

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)

            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        return top1.avg, objs.avg

    def infer(self, valid_queue, model, criterion, test_data=False):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.eval()
        device = torch.device('cuda:{}'.format(self.gpu) if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            for step, (input, target) in enumerate(valid_queue):
                input = input.to(device)
                target = target.to(device)

                logits = model(input)
                loss = criterion(logits, target)

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)

                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

        return top1.avg, objs.avg
