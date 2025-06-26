# ------------------------------------------------------------------------------------
# PointTAD 训练脚本示例
#
# 使用方法:
# 1. 根据你的GPU情况，选择下面的一个命令。
# 2. 修改 CUDA_VISIBLE_DEVICES 来指定你想使用的GPU卡号。
# 3. 确保 --nproc_per_node 的数量与你指定的GPU数量完全一致。
# ------------------------------------------------------------------------------------

# --- 多卡训练示例 ---

# 8卡训练 (原始默认配置)
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11302 --use_env main.py --dataset multithumos 

# 2卡训练 (例如，使用2号和3号GPU)
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=11302 --use_env main.py --dataset multithumos

# --- 单卡训练示例 ---
# 对于单卡训练，不需要使用 torch.distributed.launch
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset multithumos


# ------------------------------------------------------------------------------------
# Charades 数据集训练示例 (同上)
# ------------------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11302 --use_env main.py --dataset charades 

