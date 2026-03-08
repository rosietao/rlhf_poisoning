#!/bin/bash

# 训练脚本 - 15个训练任务
# epochs: 1-5
# hidden layers: 8192, 16384, 32768

accelerate launch --num_processes=1 pd_train_accelerate.py --config_file pd_train_accelerate.yaml --batch_size 128 --learning_rate 5e-5 --num_epochs 1 --hidden_layer 8192 --gpu_ids 0

accelerate launch --num_processes=1 pd_train_accelerate.py --config_file pd_train_accelerate.yaml --batch_size 128 --learning_rate 5e-5 --num_epochs 1 --hidden_layer 16384 --gpu_ids 1

accelerate launch --num_processes=1 pd_train_accelerate.py --config_file pd_train_accelerate.yaml --batch_size 128 --learning_rate 5e-5 --num_epochs 1 --hidden_layer 32768 --gpu_ids 2

accelerate launch --num_processes=1 pd_train_accelerate.py --config_file pd_train_accelerate.yaml --batch_size 128 --learning_rate 5e-5 --num_epochs 6 --hidden_layer 8192 --gpu_ids 3

accelerate launch --num_processes=1 pd_train_accelerate.py --config_file pd_train_accelerate.yaml --batch_size 128 --learning_rate 5e-5 --num_epochs 2 --hidden_layer 16384 --gpu_ids 4

accelerate launch --num_processes=1 pd_train_accelerate.py --config_file pd_train_accelerate.yaml --batch_size 128 --learning_rate 5e-5 --num_epochs 2 --hidden_layer 32768 --gpu_ids 5

accelerate launch --num_processes=1 pd_train_accelerate.py --config_file pd_train_accelerate.yaml --batch_size 128 --learning_rate 5e-5 --num_epochs 3 --hidden_layer 8192 --gpu_ids 6

accelerate launch --num_processes=1 pd_train_accelerate.py --config_file pd_train_accelerate.yaml --batch_size 128 --learning_rate 5e-5 --num_epochs 3 --hidden_layer 16384 --gpu_ids 7

accelerate launch --num_processes=1 pd_train_accelerate.py --config_file pd_train_accelerate.yaml --batch_size 128 --learning_rate 5e-5 --num_epochs 3 --hidden_layer 32768 --gpu_ids 0

accelerate launch --num_processes=1 pd_train_accelerate.py --config_file pd_train_accelerate.yaml --batch_size 128 --learning_rate 5e-5 --num_epochs 4 --hidden_layer 8192 --gpu_ids 1

accelerate launch --num_processes=1 pd_train_accelerate.py --config_file pd_train_accelerate.yaml --batch_size 128 --learning_rate 5e-5 --num_epochs 4 --hidden_layer 16384 --gpu_ids 2

accelerate launch --num_processes=1 pd_train_accelerate.py --config_file pd_train_accelerate.yaml --batch_size 128 --learning_rate 5e-5 --num_epochs 4 --hidden_layer 32768 --gpu_ids 3

accelerate launch --num_processes=1 pd_train_accelerate.py --config_file pd_train_accelerate.yaml --batch_size 128 --learning_rate 5e-5 --num_epochs 5 --hidden_layer 8192 --gpu_ids 4

accelerate launch --num_processes=1 pd_train_accelerate.py --config_file pd_train_accelerate.yaml --batch_size 128 --learning_rate 5e-5 --num_epochs 5 --hidden_layer 16384 --gpu_ids 5

accelerate launch --num_processes=1 pd_train_accelerate.py --config_file pd_train_accelerate.yaml --batch_size 128 --learning_rate 5e-5 --num_epochs 5 --hidden_layer 32768 --gpu_ids 6