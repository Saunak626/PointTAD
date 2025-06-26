'''
Helper Script to extract frames from thumos14 videos.
用于从 thumos14 视频中提取帧的辅助脚本。
(已使用多进程加速和健壮性优化)
'''
import os
import subprocess
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm


def extract_frame(video_info, video_root, frame_root, timeout=300):
    """
    为单个视频提取帧的函数（供多进程调用）。
    :param video_info: 一个元组，包含 (subset, 视频文件名)
    :param video_root: 视频根目录
    :param frame_root: 帧存放根目录
    :param timeout: 单个视频处理的超时时间（秒）
    :return: 一个元组 (状态, 视频名, 消息)
    """
    subset, vid_file = video_info
    vid_name = os.path.splitext(vid_file)[0]

    input_path = os.path.join(video_root, subset, vid_file)
    output_path = os.path.join(frame_root, subset, vid_name)

    if os.path.exists(output_path) and os.listdir(output_path):
        return 'SKIP', vid_name, "帧已存在"

    os.makedirs(output_path, exist_ok=True)

    cmd = [
        'ffmpeg', '-i', input_path,
        '-f', 'image2',
        '-vf', 'fps=10',
        '-s', '256x256',
        '-loglevel', 'error',
        f'{output_path}/img_%05d.jpg'
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        return 'SUCCESS', vid_name, "处理成功"
    except subprocess.TimeoutExpired:
        return 'TIMEOUT', vid_name, f"处理超时 ({timeout}秒)"
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode(errors='ignore').strip()
        return 'FAIL', vid_name, f"ffmpeg执行失败: {error_message}"

def main():
    # 视频文件所在的根目录
    video_path = '/home/swq/Data/thumos14_videos'
    # 提取出的帧图像要保存的根目录 (修正为项目相对路径)
    frame_path = 'data/multithumos_frames'

    subsets = ['training', 'testing']

    all_videos_to_process = []
    for subset in subsets:
        subset_video_path = os.path.join(video_path, subset)

        if not os.path.isdir(subset_video_path):
            print(f"警告：找不到目录 {subset_video_path}，将跳过。")
            continue

        os.makedirs(os.path.join(frame_path, subset), exist_ok=True)
        for vid_file in os.listdir(subset_video_path):
            all_videos_to_process.append((subset, vid_file))

    if not all_videos_to_process:
        print("没有找到需要处理的视频。")
        return
        
    num_workers = os.cpu_count()
    print(f"检测到 {num_workers} 个CPU核心，将使用它们进行并行处理...")

    process_func = partial(
        extract_frame, video_root=video_path, frame_root=frame_path, timeout=300)

    results = []
    # 采用更鲁棒的方式管理进程池和进度条
    with Pool(num_workers) as p:
        # list() 会强制等待所有任务完成，tqdm则包裹这个迭代器来显示进度
        results = list(tqdm(p.imap_unordered(process_func, all_videos_to_process), 
                            total=len(all_videos_to_process), 
                            desc="提取视频帧"))

    print("\n所有视频帧提取任务完成。")

    # 分类并报告结果
    success_count = 0
    skipped_count = 0
    failed_videos = []

    for status, vid_name, message in results:
        if status == 'SUCCESS':
            success_count += 1
        elif status == 'SKIP':
            skipped_count += 1
        else: # FAIL or TIMEOUT
            failed_videos.append((vid_name, status, message))

    print("\n--- 提取结果 ---")
    print(f"成功提取: {success_count} 个视频")
    print(f"已存在并跳过: {skipped_count} 个视频")
    print(f"失败: {len(failed_videos)} 个视频")

    if failed_videos:
        print("\n失败的视频详情:")
        for vid_name, status, message in failed_videos:
            print(f"  - 视频: {vid_name} ({status})\n    原因: {message}")


if __name__ == '__main__':
    main()

