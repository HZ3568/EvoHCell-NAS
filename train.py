import os
import json
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset

from darts import utils
from darts import genotypes
from darts.model_hetero_cell import NetworkCIFARHeteroCell as Network

parser = argparse.ArgumentParser("train_cifar10")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--min_learning_rate', type=float, default=1e-3, help='minimum learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')  # 600
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')  # 36
parser.add_argument('--layers', type=int, default=20, help='total number of layers')  # 20
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save_dir', type=str, default=None, help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default=None, help='genotype name in darts/genotypes.py')
parser.add_argument('--genotype_json', type=str, default=None,
                    help='path to a single candidate json file (e.g., candidate_0.json)')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--num_workers', type=int, default=2, help='num of data loader workers')
args = parser.parse_args()

logger = None
CIFAR_CLASSES = 10


def dict_to_genotype(gdict):
    """将 JSON 字典恢复为 Genotype 对象。"""
    return genotypes.Genotype(
        normal=[(op, int(idx)) for op, idx in gdict["normal"]],
        normal_concat=tuple(int(i) for i in gdict["normal_concat"]),
        reduce=[(op, int(idx)) for op, idx in gdict["reduce"]],
        reduce_concat=tuple(int(i) for i in gdict["reduce_concat"]),
    )


def load_genotype_list():
    """
    支持两种输入方式：
    1. --arch <name>: 从 darts/genotypes.py 加载预定义架构
    2. --genotype_json <path>: 从单个候选 JSON 文件加载（如 candidate_0.json）

    JSON 文件格式要求：
    - 顶层直接是 genotype list，或
    - 顶层包含 "genotype_list" 字段
    """
    if args.arch is not None:
        if not hasattr(genotypes, args.arch):
            raise ValueError(f"genotypes.py 中不存在架构: {args.arch}")
        genotype = getattr(genotypes, args.arch)
        return [genotype] if not isinstance(genotype, list) else genotype

    if args.genotype_json is not None:
        with open(args.genotype_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 情况1：顶层直接就是 genotype list
        if isinstance(data, list):
            return [dict_to_genotype(g) if isinstance(g, dict) else g for g in data]

        # 情况2：顶层包含 genotype_list（推荐格式，如 candidate_0.json）
        if isinstance(data, dict) and "genotype_list" in data:
            glist = data["genotype_list"]
            return [dict_to_genotype(g) if isinstance(g, dict) else g for g in glist]

        raise ValueError(
            "genotype_json 格式不正确：必须是 list，或包含 key 'genotype_list'。\n"
            "提示：请使用 search.py 生成的 candidate_*.json 文件，而不是 top_candidates.json。"
        )

    raise ValueError("You must provide either --arch or --genotype_json")


def main():
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.save_dir = utils.create_exp_dir(stage="train")
    global logger
    logger = utils.setup_logger(name="train", save_dir=args.save_dir, level="INFO")

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        cudnn.benchmark = False
        cudnn.enabled = True
        cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        logger.info('gpu device = %d', args.gpu)
    else:
        device = torch.device('cpu')
        torch.manual_seed(args.seed)
        logger.info('No gpu device available, using cpu')

    logger.info("args = %s", args)

    genotype_list = load_genotype_list()
    if len(genotype_list) != args.layers:
        raise ValueError(
            f"len(genotype_list)={len(genotype_list)} must equal args.layers={args.layers}"
        )

    model = Network(
        args.init_channels,
        CIFAR_CLASSES,
        args.layers,
        args.auxiliary,
        genotype_list
    )
    model = model.to(device)

    logger.info("param size = %fMB", utils.count_parameters_in_MB(model))
    total_params = sum(x.data.nelement() for x in model.parameters())
    logger.info('Model total parameters: %d', total_params)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_transform, valid_transform = utils.data_transforms_cifar10(
        args.cutout,
        args.cutout_length
    )

    train_data = dset.CIFAR10(
        root=args.data,
        train=True,
        download=True,
        transform=train_transform
    )
    valid_data = dset.CIFAR10(
        root=args.data,
        train=False,
        download=True,
        transform=valid_transform
    )

    pin_memory = torch.cuda.is_available()

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=args.num_workers
    )

    valid_queue = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.batch_size // 2,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=args.num_workers
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        float(args.epochs),
        eta_min=args.min_learning_rate
    )

    best_valid_acc = 0.0

    for epoch in range(args.epochs):
        current_lr = scheduler.get_last_lr()[0]
        logger.info('epoch %d lr %e', epoch, current_lr)

        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer, device)
        logger.info('train_loss %.4f train_acc %.2f', train_obj, train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion, device)
        logger.info('valid_loss %.4f valid_acc %.2f', valid_obj, valid_acc)

        scheduler.step()

        utils.save(model, os.path.join(args.save_dir, 'weights.pt'))

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            utils.save(model, os.path.join(args.save_dir, 'best_weights.pt'))

    logger.info('best_valid_acc %f', best_valid_acc)


def train(train_queue, model, criterion, optimizer, device):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(input)
        if isinstance(outputs, tuple):
            logits, logits_aux = outputs
            loss = criterion(logits, target)
            if args.auxiliary:
                loss_aux = criterion(logits_aux, target)
                loss = loss + args.auxiliary_weight * loss_aux
        else:
            logits = outputs
            loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logger.info('train step=%03d loss=%.4f top1=%.2f top5=%.2f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, device):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            outputs = model(input)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logger.info('valid step=%03d loss=%.4f top1=%.2f top5=%.2f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


# 使用示例：
# python train.py --genotype_json ./results/search_xxx/candidate_0.json

if __name__ == '__main__':
    main()
