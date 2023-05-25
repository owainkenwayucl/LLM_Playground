#!/bin/bash -l

#$ -l mem=100G
#$ -l h_rt=24:0:0
#$ -l tmpfs=50G
#$ -cwd

set -e

LLAMA_PATH=${LLAMA_PATH:-"${HOME}/Scratch/llama"}

module load openssl/1.1.1t python/3.11.3

virtualenv vicuna_runtume
source vicuna_runtime/bin/activate

pip install fschat

python3 -m fastchat.model.apply_delta \
    --base-model-path ${LLAMA_PATH}/7B \
    --target-model-path vicuna-7b \
    --delta-path lmsys/vicuna-7b-delta-v1.1

python3 -m fastchat.model.apply_delta \
    --base-model-path ${LLAMA_PATH}/llama-13b \
    --target-model-path vicuna-13b \
    --delta-path lmsys/vicuna-13b-delta-v1.1
