#!/bin/bash -l

set -e

eval "$(conda shell.bash hook)"

python -m conda create -n sd

python -m conda activate sd

python -m conda install python=3.11.5

pip install --upgradep pip torch torchvision diffusers transformers accelerate safetensors ipykernel

python -m ipykernel install --user --name=stablediffusion