#!/bin/bash
#$ -cwd
#$ -wd /data/home/acw773/Deformable-3D-Gaussians
#$ -j y
#$ -pe smp 12          # 12 cores per GPU
#$ -l h_rt=2:59:0    # 240 hours runtime
#$ -l gpu=1           # request 1 GPU
#$ -o logs/
#$ -m ea
#$ -l gpu_type=ampere


# cd Deformable-3D-Gaussians

module load gcc/6.3.0
module load cuda/11.6.2
export CUDA_HOME=/share/apps/centos7/cuda/11.6.2
export TORCH_CUDA_ARCH_LIST="8.0"
module load anaconda3
conda activate deformable_gaussian_env



# python train.py -s /data/scratch/acw773/HO3D_v2/train/ABF10 -m output/ho3d/check-need-delete --eval --iterations 40000

# python train_HOI_sem.py -s /data/scratch/acw773/HO3D_v2/train/ABF10 -m output/ho3d/just_test --motion_offset_flag --motion_flag --smpl_type mano --actor_gender right  --eval --iterations 40000

# python train_HOI_def.py -s /data/scratch/acw773/HO3D_v2/train/ABF10 -m output/ho3d/d_xyz_only_obj --motion_offset_flag --motion_flag --smpl_type mano --actor_gender right  --eval --iterations 40000

# 第一个实验室，scale加了，xyz只有物加了
# 上一个实验是: scale， xyz，手物都加；
# 这个实验室，scale，xyz都只有物加
# python train_HOI.py -s /data/scratch/acw773/HO3D_v2/train/ABF10 -m output/ho3d/only_obj --motion_offset_flag --motion_flag --smpl_type mano --actor_gender right  --eval --iterations 40000


# python train_HOI_nomutual.py -s /data/scratch/acw773/HO3D_v2/train/ABF10 -m output/ho3d/nomutual --motion_offset_flag --motion_flag --smpl_type mano --actor_gender right  --eval --iterations 40000

# python train_HOI_nomutual.py -s /data/scratch/acw773/HO3D_v2/train/ABF10 -m output/ho3d/nomutual-save-rot --motion_offset_flag --motion_flag --smpl_type mano --actor_gender right  --eval --iterations 40000


# python train_HOI_sem.py -s /data/scratch/acw773/HO3D_v2/train/ABF10 -m output/ho3d/sem --motion_offset_flag --motion_flag --smpl_type mano --actor_gender right  --eval --iterations 40000



python train_HOI.py -s /data/scratch/acw773/HO3D_v2/train/ABF10 -m output/ho3d/mutual --motion_offset_flag --motion_flag --smpl_type mano --actor_gender right  --eval --iterations 40000



# python render.py -m output/ho3d/test --mode render
# python metrics.py -m output/ho3d/test










# python train_HOI.py -s /data/scratch/acw773/HO3D_v2/train/ABF10 -m output/ho3d/just_test --motion_offset_flag --motion_flag --smpl_type mano --actor_gender right  --eval --iterations 40000
