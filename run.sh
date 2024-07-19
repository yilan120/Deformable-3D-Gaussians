#!/bin/bash
#$ -cwd
#$ -wd /data/home/acw773/GauHuman
#$ -j y
#$ -pe smp 12          # 12 cores per GPU
#$ -l h_rt=0:59:0    # 240 hours runtime
#$ -l gpu=1           # request 1 GPU
#$ -o logs/
#$ -m ea
#$ -l gpu_type=ampere


module load gcc/6.3.0
module load cuda/11.6.2
export CUDA_HOME=/share/apps/centos7/cuda/11.6.2
export TORCH_CUDA_ARCH_LIST="8.0"
module load anaconda3
conda activate deformable_gaussian_env



# python train.py -s path/to/your/d-nerf/dataset -m output/exp-name --eval --is_blender
python train.py -s /data/scratch/acw773/HO3D_v2/train/ABF10 -m output/ho3d/check-need-delete --eval --iterations 40000


# python render.py -m output/ho3d/test --mode render
# python metrics.py -m output/ho3d/test