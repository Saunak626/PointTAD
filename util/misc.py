# Copy and paste from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import datetime
import os
import pickle
import subprocess
import time
from collections import defaultdict, deque
from typing import List, Optional

import torch
import torch.distributed as dist
# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
from torch import Tensor

class SmoothedValue(object):
    """
    追踪一系列数值，并提供对窗口内平滑值或全局平均值的访问。
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = '{median:.4f} ({global_avg:.4f})'
        # 使用双端队列存储最近的数值
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """更新数值"""
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        在多进程之间同步total和count的值。
        警告: 不会同步deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64,
                         device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """计算当前窗口内的中位数"""
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """计算当前窗口内的平均值"""
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """计算全局平均值"""
        return self.total / self.count

    @property
    def max(self):
        """获取当前窗口内的最大值"""
        return max(self.deque)

    @property
    def value(self):
        """获取最近添加的数值"""
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median,
                               avg=self.avg,
                               global_avg=self.global_avg,
                               max=self.max,
                               value=self.value)


def all_gather(data):
    """
    在所有进程上运行all_gather，支持任意可pickle序列化的数据。
    Args:
        data: 任何可pickle序列化的对象
    Returns:
        list[data]: 从每个rank收集到的数据列表
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # 将数据序列化为Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to('cuda')

    # 获取每个rank上的Tensor大小
    local_size = torch.tensor([tensor.numel()], device='cuda')
    size_list = [torch.tensor([0], device='cuda') for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # 从所有rank接收Tensor
    # 因为torch.all_gather不支持不同形状的张量，所以需要填充
    tensor_list = []
    for _ in size_list:
        tensor_list.append(
            torch.empty((max_size, ), dtype=torch.uint8, device='cuda'))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size, ),
                              dtype=torch.uint8,
                              device='cuda')
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    将字典中的所有值从所有进程中reduce。
    Args:
        input_dict (dict): 将要被reduce的所有值
        average (bool): 是否进行平均或求和
    Returns:
        一个与input_dict字段相同，但经过reduce操作的字典。
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # 对键进行排序，以确保在所有进程中顺序一致
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    """指标记录器，用于记录和打印训练过程中的各种指标。"""
    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """
        优化日志输出格式，使其更简洁、易读。
        """
        loss_str_parts = []
        
        # 1. 优先显示和高亮核心指标
        if "loss" in self.meters:
            # 使用ANSI转义码为 'loss' 添加红色粗体效果
            loss_str_parts.append(f"\033[1;31mloss: {str(self.meters['loss'])}\033[0m")
        if "class_error" in self.meters:
            # 为 'class_error' 添加品红色
            loss_str_parts.append(f"\033[95mclass_error: {str(self.meters['class_error'])}\033[0m")
        if "lr" in self.meters:
            # 为 'lr' 添加黄色
            loss_str_parts.append(f"\033[93mlr: {str(self.meters['lr'])}\033[0m")

        # 2. 将主要的损失分量分组显示，并过滤掉不重要的细节
        detail_losses = []
        # 对键进行排序以保证顺序一致
        for name in sorted(self.meters.keys()):
            # 只选择主要的、加权的损失分量进行展示
            if name in ['loss_ce', 'loss_giou', 'loss_seg', 'loss_dense']:
                meter = self.meters[name]
                # 只显示中位数，不显示全局平均值，使日志更紧凑
                detail_losses.append(f"{name.split('_')[-1]}: {meter.median:.4f}")
        
        if detail_losses:
            loss_str_parts.append("loss_details: [" + " | ".join(detail_losses) + "]")
            
        return self.delimiter.join(loss_str_parts)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """每隔一定迭代次数打印日志。"""
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}',
                'time: {time}', 'data: {data}', 'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}',
                'time: {time}', 'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(
                        log_msg.format(i,
                                       len(iterable),
                                       eta=eta_string,
                                       meters=str(self),
                                       time=str(iter_time),
                                       data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    """获取当前git仓库的commit SHA、状态和分支信息。"""
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command,
                                       cwd=cwd).decode('ascii').strip()

    sha = 'N/A'
    diff = 'clean'
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = 'has uncommited changes' if diff else 'clean'
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f'sha: {sha}, status: {diff}, branch: {branch}'
    return message


def collate_fn(batch):
    """自定义的collate_fn，用于DataLoader。"""
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    """沿每个轴计算列表的最大值。"""
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    """
    从一个张量列表构建一个NestedTensor。
    NestedTensor包含一个批次的张量和一个对应的掩码。
    """
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() 不能很好地导出到 ONNX
            # 调用 _onnx_nested_tensor_from_tensor_list()
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO: 使其支持不同尺寸的图像
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() 是
# nested_tensor_from_tensor_list() 的一个实现，
# 支持ONNX追踪。
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list):
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list
                         ]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(
            img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m,
                                              (0, padding[2], 0, padding[1]),
                                              'constant', 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


class NestedTensor(object):
    """用于将不同大小的张量打包在一起的辅助类。"""
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def setup_for_distributed(is_master):
    """
    此函数在非主进程中禁用打印。
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """检查分布式环境是否可用并已初始化。"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """获取分布式环境中的进程总数。"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """获取当前进程的rank。"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """判断当前进程是否为主进程（rank 0）。"""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """只在主进程上保存。"""
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """初始化分布式训练模式。"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url),
          flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend,
                                         init_method=args.dist_url,
                                         world_size=args.world_size,
                                         rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1, )):
    """根据指定的k值计算precision@k。"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
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


def interpolate(input,
                size=None,
                scale_factor=None,
                mode='nearest',
                align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    等同于nn.functional.interpolate，但支持空的batch size。
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(input, size, scale_factor,
                                                   mode, align_corners)

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor,
                                                mode, align_corners)
