#!/bin/sh

cache_dir=/home/ppillai6/Desktop/BERT_training/.cache
model_save_dir=/home/ppillai6/Desktop/BERT_training/model_save

XRT_TPU_CONFIG="tpu_worker;0;10.27.135.26:8470" python -u geoscibert_pretrain.py --model_save_dir $model_save_dir --cache_dir $cache_dir