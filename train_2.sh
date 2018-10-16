#!/bin/sh
DATASET_DIR=/mogu/liubang/mytf/tfrecords
TRAIN_DIR=./logs2/
NUM=7
CHECKPOINT_PATH=./logs/model.ckpt-73999
python -u train_ssd_network.py \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --num_classes=${NUM} \
    --dataset_split_name=train \
    --model_name=ssd_512_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=120 \
    --save_interval_secs=1200 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.0001 \
    --batch_size=12 \
    --train_dir=${TRAIN_DIR} \
    --gpu_memory_fraction=0.7
