# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Modified from RTD-Net (https://github.com/MCG-NJU/RTD-Action)

PointTAD Training and Inference functions.

"""

import json
import math
import sys
from typing import Iterable

import torch
from termcolor import colored

import util.misc as utils
from datasets.evaluate import Evaluator

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    args,
                    postprocessors=None):
    """
    对模型进行一个 epoch 的训练。
    :param model: 需要训练的模型。
    :param criterion: 损失函数计算模块。
    :param data_loader: 训练数据加载器。
    :param optimizer: 优化器。
    :param device: 计算设备 (CPU or GPU)。
    :param epoch: 当前的 epoch 序号。
    :param args: 命令行参数。
    :param postprocessors: 后处理器 (训练时通常为None)。
    :return: 一个包含平均指标的字典，以及最后一个batch的损失字典。
    """
    # 将模型和损失函数设置为训练模式
    model.train()
    criterion.train()
    # 初始化指标记录器，用于跟踪和打印日志
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    metric_logger.add_meter(
        'class_error', utils.SmoothedValue(window_size=1,
                                            fmt='{value:.2f}'))
    
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 30

    max_norm = args.clip_max_norm

    # 遍历数据加载器中的所有批次数据
    for vid_name_list, locations, x, targets, num_frames, base \
        in metric_logger.log_every(data_loader, print_freq, header):

        # 将输入数据和目标移动到指定的设备
        x = x.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # 模型前向传播，得到输出
        outputs = model(x)
        # 计算损失
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        # 根据权重计算加权总损失
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys()
                     if k in weight_dict)
        n_parameters = sum(p.numel() for p in model.parameters())
        losses += 0 * n_parameters
        # 在多卡训练中，聚合所有进程的损失
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # 未加权的损失值 (用于日志记录)
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v
            for k, v in loss_dict_reduced.items()
        }
        # 加权后的损失值 (用于日志记录)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        # 检查损失值是否有效
        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        # 进行梯度裁剪，防止梯度爆炸
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # 更新模型权重
        optimizer.step()

        # 更新日志记录器中的各项指标
        metric_logger.update(loss=loss_value,
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    # 在所有进程之间同步指标的全局平均值
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg
            for k, meter in metric_logger.meters.items()}, loss_dict

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, args):
    print(colored('evaluate', 'red'))
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter(
        'class_error', utils.SmoothedValue(window_size=1,
                                            fmt='{value:.2f}'))
    header = 'Test:'

    evaluator = Evaluator()

    video_pool = list(load_json(args.annotation_path).keys())
    video_pool.sort()
    video_dict = {i: video_pool[i] for i in range(len(video_pool))}
    print_freq = 30
    
    for vid_name_list, locations, x, targets, num_frames, base in metric_logger.log_every(
            data_loader, print_freq, header):
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(x)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v
            for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        results, dense_results = postprocessors['results'](outputs, num_frames, base)

        for target, output, dense_res, base_loc in zip(targets, results, dense_results, base):
            vid = video_dict[target['video_id'].item()]
            dense_gt = target['dense_gt']
            if args.dense_result:
                torch.save(dense_res, f'dense_results/{vid}_{base_loc}_dense')
            evaluator.update(vid, output, base_loc)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    evaluator.synchronize_between_processes()
    print('Averaged stats:', metric_logger)

    return evaluator, loss_dict
