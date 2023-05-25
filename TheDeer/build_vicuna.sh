#!/bin/bash -l

#$ -l mem=100G
#$ -l h_rt=24:0:0
#$ -l tmpfs=50G
#$ -cwd

set -e

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

LLAMA_PATH=${LLAMA_PATH:-"${HOME}/Scratch/llama"}

module load openssl/1.1.1t python/3.11.3

rm -rf vicuna_runtime
virtualenv vicuna_runtime
source vicuna_runtime/bin/activate

pip install fschat

python3 -m transformers.models.llama.convert_llama_weights_to_hf \
    --input_dir ${LLAMA_PATH} --model_size 7B --output_dir ${LLAMA_PATH}/7B_hf

python3 -m transformers.models.llama.convert_llama_weights_to_hf \
    --input_dir ${LLAMA_PATH} --model_size 13B --output_dir ${LLAMA_PATH}/13B_hf

python3 -m fastchat.model.apply_delta \
    --base-model-path ${LLAMA_PATH}/7B_hf \
    --target-model-path vicuna-7b \
    --delta-path lmsys/vicuna-7b-delta-v1.1

python3 -m fastchat.model.apply_delta \
    --base-model-path ${LLAMA_PATH}/13B_hf \
    --target-model-path vicuna-13b \
    --delta-path lmsys/vicuna-13b-delta-v1.1
