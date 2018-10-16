#!/bin/sh

DATASET_DIR=/mogu/liubang/mytf/tfrecords
TRAIN_DIR=./logs2/ #模型的目录
NUM=7
EVAL_DIR=./eval/ #评估的结果
python -u eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --num_classes=${NUM} \
    --dataset_split_name=val \
    --model_name=ssd_512_vgg \
    --checkpoint_path=${TRAIN_DIR} \
    --wait_for_checkpoints=False \
    --batch_size=10 \
    --max_num_batches=1000
