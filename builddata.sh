#!/bin/sh
SOURCE=VOCclothsneg
FILENAME=train.txt
STATE=train
DATASET_DIR=/mogu/liubang/mytf/${SOURCE}/
OUTPUT_DIR=/mogu/liubang/mytf/tfrecords/
python tf_convert_data.py \
        --dataset_name=${FILENAME} \
        --dataset_dir=${DATASET_DIR} \
        --output_name=${SOURCE}_${STATE} \
        --output_dir=${OUTPUT_DIR}

SOURCE=VOCclothsneg
FILENAME=train_hard.txt
STATE=train
DATASET_DIR=/mogu/liubang/mytf/${SOURCE}/
OUTPUT_DIR=/mogu/liubang/mytf/tfrecords/
python tf_convert_data.py \
        --dataset_name=${FILENAME} \
        --dataset_dir=${DATASET_DIR} \
        --output_name=${SOURCE}_hard_${STATE} \
        --output_dir=${OUTPUT_DIR}

SOURCE=VOCclothsonepiece
FILENAME=train_bag_shoes_hard_onepiece.txt
STATE=train
DATASET_DIR=/mogu/liubang/mytf/${SOURCE}/
OUTPUT_DIR=/mogu/liubang/mytf/tfrecords/
python tf_convert_data.py \
        --dataset_name=${FILENAME} \
        --dataset_dir=${DATASET_DIR} \
        --output_name=${SOURCE}_${STATE} \
        --output_dir=${OUTPUT_DIR}

SOURCE=VOCclothsonepiece
FILENAME=val.txt
STATE=val
DATASET_DIR=/mogu/liubang/mytf/${SOURCE}/
OUTPUT_DIR=/mogu/liubang/mytf/tfrecords/
python tf_convert_data.py \
        --dataset_name=${FILENAME} \
        --dataset_dir=${DATASET_DIR} \
        --output_name=${SOURCE}_${STATE} \
        --output_dir=${OUTPUT_DIR}

SOURCE=VOCclothsnormal
FILENAME=train_bag_shoes_hard_onepiece.txt
STATE=train
DATASET_DIR=/mogu/liubang/mytf/${SOURCE}/
OUTPUT_DIR=/mogu/liubang/mytf/tfrecords/
python tf_convert_data.py \
        --dataset_name=${FILENAME} \
        --dataset_dir=${DATASET_DIR} \
        --output_name=${SOURCE}_${STATE} \
        --output_dir=${OUTPUT_DIR}

SOURCE=VOCclothsnormal
FILENAME=val.txt
STATE=val
DATASET_DIR=/mogu/liubang/mytf/${SOURCE}/
OUTPUT_DIR=/mogu/liubang/mytf/tfrecords/
python tf_convert_data.py \
        --dataset_name=${FILENAME} \
        --dataset_dir=${DATASET_DIR} \
        --output_name=${SOURCE}_${STATE} \
        --output_dir=${OUTPUT_DIR}

SOURCE=VOCclothswuliu
FILENAME=train_bag_shoes_hard_onepiece.txt
STATE=train
DATASET_DIR=/mogu/liubang/mytf/${SOURCE}/
OUTPUT_DIR=/mogu/liubang/mytf/tfrecords/
python tf_convert_data.py \
        --dataset_name=${FILENAME} \
        --dataset_dir=${DATASET_DIR} \
        --output_name=${SOURCE}_${STATE} \
        --output_dir=${OUTPUT_DIR}

SOURCE=VOCclothswuliu
FILENAME=val.txt
STATE=val
DATASET_DIR=/mogu/liubang/mytf/${SOURCE}/
OUTPUT_DIR=/mogu/liubang/mytf/tfrecords/
python tf_convert_data.py \
        --dataset_name=${FILENAME} \
        --dataset_dir=${DATASET_DIR} \
        --output_name=${SOURCE}_${STATE} \
        --output_dir=${OUTPUT_DIR}
