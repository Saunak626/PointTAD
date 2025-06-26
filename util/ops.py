"""
Helper functions for Temporal Action Detection.
用于时序动作检测的辅助函数。
"""
import torch
import torch.nn.functional as F
import numpy as np

def bilinear_sampling(value, sampling_locations):
    """
    进行双线性采样。
    :param value: (N, T, N_heads, Dim) 输入的值
    :param sampling_locations: (N, N_query, N_heads, N_level, N_points, 2) 采样点位置
    :return: 采样后的值
    """
    # values: N, T, N_heads, Dim
    # sampling_locations: N, N_query, N_heads, N_level, N_points, 2
    N_, T, n_heads, D_ = value.shape
    _, Lq_, n_heads, L_, P_, _ = sampling_locations.shape
    # 将采样位置从[0, 1]归一化到[-1, 1]以适应grid_sample
    sampling_grids = 2 * sampling_locations - 1
    lid_ = 0
    H_ = 1
    W_ = T
    # 调整value的维度以适应grid_sample的输入格式
    value_l_ = value.permute(0,2,3,1).reshape(N_*n_heads, 1, D_, H_, W_).repeat(1,Lq_,1,1,1)
    value_l_ = value_l_.flatten(0,1)
    # 调整采样网格的维度
    sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
    sampling_grid_l_ = sampling_grid_l_.flatten(0,1).unsqueeze(-3)
    # 执行双线性插值采样
    sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                        mode='bilinear', padding_mode='zeros', align_corners=True)
    output = sampling_value_l_
    return output.contiguous()

def convert(points):
    """
    将一系列点转换为表示段的起始和结束点。
    :param points: (N, nr_segments, num_querypoints) 或 (N * nr_segments, num_querypoints) 输入的点
    :return: (N, nr_segments, 2) 或 (N * nr_segments, 2) 表示段的[开始, 结束]
    """
    # input : (N, nr_segments, num_querypoints) or (N * nr_segments, num_querypoints)
    # output: (N, nr_segments, 2) or (N * nr_segments, 2)
    if len(points.shape) == 3:
        N, nr_segments, num_querypoints = points.shape
        pred_segments = points.new_zeros((N, nr_segments, 2))
        # 使用除了最后三分之一查询点以外的点来确定段的边界
        # 开始时间是这些点的最小值
        pred_segments[:,:,0] = torch.min(points[:,:,:-num_querypoints//3], dim=-1)[0]
        # 结束时间是这些点的最大值
        pred_segments[:,:,1] = torch.max(points[:,:,:-num_querypoints//3], dim=-1)[0]
        return pred_segments
    elif len(points.shape) == 2:
        N_nr_segments, num_querypoints = points.shape
        pred_segments = points.new_zeros((N_nr_segments, 2))
        pred_segments[:,0] = torch.min(points[:,:-num_querypoints//3], dim=-1)[0]
        pred_segments[:,1] = torch.max(points[:,:-num_querypoints//3], dim=-1)[0]
        return pred_segments
    else:
        print("Wrong Input in Convert!")
        return

# rewrite for temporal localization setting
def prop_cl_to_se(x):
    """
    将 [中心点, 长度] 格式的提议转换为 [开始, 结束] 格式。
    :param x: (..., 2) 输入提议，最后一维是 [中心点, 长度]
    :return: (..., 2) 输出提议，最后一维是 [开始, 结束]，值被限制在[0, 1]之间
    """
    c, l = x.unbind(-1)
    b = [(c - 0.5 * l), (c + 0.5 * l)]
    return torch.stack(b, dim=-1).clamp(0, 1)


def prop_se_to_cl(x):
    """
    将 [开始, 结束] 格式的提议转换为 [中心点, 长度] 格式。
    :param x: (..., 2) 输入提议，最后一维是 [开始, 结束]
    :return: (..., 2) 输出提议，最后一维是 [中心点, 长度]
    """
    s, e = x.unbind(-1)
    b = [(s + e) / 2, (e - s)]
    return torch.stack(b, dim=-1)


def prop_relative_to_absolute(x, base, window_size, interval):
    """
    将相对的段坐标转换为绝对坐标。
    :param x: (..., 2) 相对的 [开始, 结束] 坐标
    :param base: 基准位置
    :param window_size: 窗口大小
    :param interval: 间隔
    :return: (..., 2) 绝对的 [开始, 结束] 坐标
    """
    s, e = x.unbind(-1)
    num_samples = s.shape[1]
    base = base.unsqueeze(1).repeat(1, num_samples).cuda()
    b = [s * window_size * interval + base, e * window_size * interval + base]
    return torch.stack(b, dim=-1)


def segment_tiou(seg_a, seg_b):
    """
    计算两组时间段之间的 tIoU (temporal Intersection over Union)。
    :param seg_a: (N, 2) 第一组时间段 [开始, 结束]
    :param seg_b: (M, 2) 第二组时间段 [开始, 结束]
    :return: (N, M) tIoU 矩阵
    """
    # gt: [N, 2], detections: [M, 2]
    N = seg_a.shape[0]
    M = seg_b.shape[0]

    tiou = torch.zeros((N, M)).to(seg_a.device)
    for i in range(N):
        # 计算交集的结束点和开始点
        inter_max_xy = torch.min(seg_a[i, 1], seg_b[:, 1])
        inter_min_xy = torch.max(seg_a[i, 0], seg_b[:, 0])

        # 计算交集长度，如果无交集则为0
        inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)

        # 计算并集长度
        union = (seg_b[:, 1] - seg_b[:, 0]) + (seg_a[i, 1] -
                                               seg_a[i, 0]) - inter

        # 计算 tIoU
        tiou[i, :] = inter / (union+1e-8)

    return tiou  # (N, M)
