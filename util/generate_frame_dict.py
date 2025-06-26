'''
Helper Script to generate 'number of frames' dictionary for extracted frames.
为提取的帧生成"帧数"字典的辅助脚本。
(已支持断点执行)
'''
import os 
import json
from tqdm import tqdm

def load_json(file):
    """
    加载并解析JSON文件。如果文件不存在，返回一个空字典。
    :param file: JSON文件路径
    :return: 解析后的数据或空字典
    """
    if not os.path.exists(file):
        return {}
    try:
        with open(file, 'r') as json_file:
            data = json.load(json_file)
            return data
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

# 帧数信息JSON文件的保存路径
output_json_path = 'datasets/multithumos_frames.json'

# 1. 加载已有的帧数记录，实现断点续处理
frame_dict = load_json(output_json_path)
print(f"已加载 {len(frame_dict)} 条现存的视频帧数记录。")

# 存储提取出的视频帧的根目录
frame_path = 'data/multithumos_frames' 
# 定义要处理的数据子集
subsets = ['training','testing']

# 2. 遍历帧文件夹，只处理新视频
for subset in subsets:
    subset_frame_path = os.path.join(frame_path, subset)
    if not os.path.isdir(subset_frame_path):
        print(f"警告：找不到目录 {subset_frame_path}，将跳过。")
        continue

    # 获取该子集下所有视频（即帧文件夹）的列表
    video_list = os.listdir(subset_frame_path)
    
    print(f"正在检查 {subset} 子集的 {len(video_list)} 个视频...")
    for vid in tqdm(video_list, desc=f"处理 {subset}"):
        # 如果视频记录已存在，则跳过
        if vid in frame_dict:
            continue
        
        try:
            # 计算该视频目录下的文件数量，即帧数
            num_frames = len(os.listdir(os.path.join(subset_frame_path, vid)))
            # 将视频ID和对应的帧数存入字典
            frame_dict[vid] = num_frames
        except FileNotFoundError:
            print(f"警告：在处理 {vid} 时找不到对应的帧文件夹，已跳过。")


# 3. 将更新后的字典写回文件
with open(output_json_path, 'w') as f:
    # 将帧数字典以格式化的方式写入JSON文件
    # sort_keys=True: 按键排序
    # indent=4: 使用4个空格进行缩进
    # separators=(',', ': '): 定义分隔符
    json.dump(frame_dict, sort_keys=True, indent=4, separators=(',', ': '), fp=f)

print(f"\n处理完成。总共记录了 {len(frame_dict)} 个视频的帧数信息到 {output_json_path}")
