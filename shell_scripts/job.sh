#!/bin/bash
#SBATCH -c 8 
#SBATCH -p general
#SBATCH -t 1-00:00:00
#SBATCH --mem=40G
#SBATCH -G a100:1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo

python main_classification.py --data_set CheXpert --model swin_base --init ark --pretrained_weights /home/nyarava/ARK/Ark/ark6_teacher_ep200_swinb_projector1376_mlp.pth.tar --data_dir /scratch/nyarava/hey.zip/chexpertchestxrays-u20210408/ --train_list /scratch/nyarava/hey.zip/chexpertchestxrays-u20210408/CheXpert-v1.0/train.csv --val_list /scratch/nyarava/hey.zip/chexpertchestxrays-u20210408/CheXpert-v1.0/valid.csv --test_list /scratch/nyarava/hey.zip/chexpertchestxrays-u20210408/CheXpert-v1.0/test_labels.csv --lr 0.001 --opt adamw --weight-decay 1e-4 --epochs 8 --warmup-epochs 3 --batch_size 64 --sched cosine