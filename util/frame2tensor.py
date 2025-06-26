'''
Helper Script to transform raw RGB frames into image tensors.
将原始RGB帧转换为图像张量的辅助脚本。
(已使用多进程加速和健壮性优化)
'''
import os
import numpy as np
import cv2
import torch
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def process_video_to_tensor(vid_info, frame_root, tensor_root, dataset_name, pbar_position=0):
    """
    将单个视频的帧文件夹转换为一个张量文件。
    :param vid_info: 一个元组，包含 (split, vid_folder_name)
    :param frame_root: 帧文件夹的根目录
    :param tensor_root: 输出张量的根目录
    :param dataset_name: 数据集名称 ('multithumos' or 'charades')
    """
    split, vid = vid_info
    
    tensor_save_path = os.path.join(tensor_root, split, f"{vid}.pt")
    frame_folder_path = os.path.join(frame_root, split, vid)

    # 如果张量文件已存在，则跳过
    if os.path.isfile(tensor_save_path):
        return 'SKIP', vid, f"{vid}.pt 已存在"

    try:
        if not os.path.isdir(frame_folder_path):
            return 'FAIL', vid, f"帧文件夹未找到: {frame_folder_path}"

        # 获取视频的帧数量
        frame_files = sorted(os.listdir(frame_folder_path))
        num_frames = len(frame_files)
        if num_frames == 0:
            return 'FAIL', vid, "帧文件夹为空"
        
        if dataset_name == 'multithumos':
            # 直接使用排序后的文件名列表构建路径
            img_stacked_paths = [os.path.join(frame_folder_path, f) for f in frame_files]
        else: # charades
            # charades的命名格式可能不同，这里保持了原逻辑但改为使用排序后的列表
            img_stacked_paths = [os.path.join(frame_root, vid, f) for f in frame_files]

        img_list = []
        for img_path in img_stacked_paths:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return 'FAIL', vid, f"无法读取图像: {img_path}"
            img_list.append(img)
        
        # 将图像列表堆叠成一个NumPy数组
        img_array = np.stack(img_list, axis=0)
        # 将NumPy数组转换为PyTorch张量
        img_tensor = torch.from_numpy(img_array)
        
        # 保存张量到文件
        torch.save(img_tensor, tensor_save_path)
        
        # 使用tqdm.write安全地打印信息
        tqdm.write(f"[SUCCESS] {vid} -> {img_tensor.shape}")
        return 'SUCCESS', vid, "处理成功"

    except Exception as e:
        return 'FAIL', vid, f"发生未知错误: {str(e)}"

def main():
    # 指定要处理的数据集名称 ('multithumos' 或 'charades')
    dataset = 'multithumos' # or charades

    if dataset == 'multithumos':
        splits = ['training','testing']
        frame_folder_path = 'data/multithumos_frames/'
        tensor_path = 'data/multithumos_tensors/'
    else: # charades
        splits = ['']
        frame_folder_path = 'data/charades_v1_rgb/'
        tensor_path = 'data/charades_v1_rgb_tensors/'

    # 1. 准备所有需要处理的视频列表
    all_vids_to_process = []
    for split in splits:
        split_frame_path = os.path.join(frame_folder_path, split)
        os.makedirs(os.path.join(tensor_path, split), exist_ok=True)
        
        if not os.path.isdir(split_frame_path):
            print(f"警告：找不到帧目录 {split_frame_path}，将跳过。")
            continue
            
        vid_list = os.listdir(split_frame_path)
        for vid in vid_list:
            all_vids_to_process.append((split, vid))

    if not all_vids_to_process:
        print("没有找到需要处理的视频帧文件夹。")
        return

    # 2. 使用多进程池进行处理
    # 对于内存密集型任务，需要限制worker数量以防止内存溢出。
    # 4个worker是一个比较安全和通用的起始值。
    MAX_WORKERS = 4
    num_workers = min(os.cpu_count(), MAX_WORKERS)
    print(f"检测到 {os.cpu_count()} 个CPU核心，将使用 {num_workers} 个进行并行处理以优化内存使用...")

    process_func = partial(
        process_video_to_tensor, 
        frame_root=frame_folder_path, 
        tensor_root=tensor_path,
        dataset_name=dataset)

    results = []
    with Pool(num_workers) as p:
        results = list(tqdm(p.imap_unordered(process_func, all_vids_to_process),
                            total=len(all_vids_to_process),
                            desc="转换帧为张量"))
                            
    # 3. 报告结果
    success_count = 0
    skipped_count = 0
    failed_vids = []

    for status, vid_name, message in results:
        if status == 'SUCCESS':
            success_count += 1
        elif status == 'SKIP':
            skipped_count += 1
        else: # FAIL
            failed_vids.append((vid_name, message))

    print("\n--- 转换完成 ---")
    print(f"成功转换: {success_count} 个视频")
    print(f"已存在并跳过: {skipped_count} 个视频")
    print(f"失败: {len(failed_vids)} 个视频")

    if failed_vids:
        print("\n失败的视频详情:")
        for vid_name, message in failed_vids:
            print(f"  - 视频: {vid_name}\n    原因: {message}")

if __name__ == '__main__':
    main()
