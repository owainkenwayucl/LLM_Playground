#!/bin/bash -l
# git clone https://github.com/owainkenwayucl/LLM_Playground.git

eval "$(conda shell.bash hook)"

python3 -m conda create -n sd

conda activate sd
set -e 

conda install python=3.11.5

pip3 install --upgrade pip torch torchvision diffusers transformers accelerate safetensors ipykernel nvitop

python3 -m ipykernel install --user --name=stablediffusion

#export CUDA_VISIBLE_DEVICES=0,1