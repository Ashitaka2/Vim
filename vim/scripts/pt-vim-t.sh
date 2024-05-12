#!/bin/bash
# conda activate <your_env>
# cd vim;

MODEL_FLAGS="--model vim_tiny_patch8_depth24_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
TRAIN_FLAGS="--batch-size 128 --drop-path 0.0 --weight-decay 0.1 --num_workers 10"
NUM_GPUS=8
# CUDA_VISIBLE_DEVICES = 0,1,2,3,

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --use_env main.py --data-path /home/jaesungpark/data2/Vim/imgnet --output_dir output/vim_tiny/patch8/depth24 --no_amp $TRAIN_FLAGS $MODEL_FLAGS