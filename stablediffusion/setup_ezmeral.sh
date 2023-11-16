#!/bin/bash -l

set -e

eval "$(conda shell.bash hook)"

python3 -m conda create -n sd

conda activate sd

conda install python=3.11.5

pip3 install --upgrade pip torch torchvision diffusers transformers accelerate safetensors ipykernel nvitop

python3 -m ipykernel install --user --name=stablediffusion