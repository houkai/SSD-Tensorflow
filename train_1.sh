#!/bin/sh
DATASET_DIR=/mogu/liubang/mytf/tfrecords
TRAIN_DIR=./logs/
NUM=7
CHECKPOINT_PATH=./checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt
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
    --learning_rate=0.001 \
    --batch_size=16 \
    --train_dir=${TRAIN_DIR} \
    --checkpoint_exclude_scopes=ssd_512_vgg/block4_box,ssd_512_vgg/block7_box,ssd_512_vgg/block8_box,ssd_512_vgg/block9_box,ssd_512_vgg/block10_box,ssd_512_vgg/block11_box,ssd_512_vgg/block12_box \
    --gpu_memory_fraction=0.6 \
    --trainable_scopes=ssd_512_vgg/block4_box,ssd_512_vgg/block7_box,ssd_512_vgg/block8_box,ssd_512_vgg/block9_box,ssd_512_vgg/block10_box,ssd_512_vgg/block11_box,ssd_512_vgg/block12_box

