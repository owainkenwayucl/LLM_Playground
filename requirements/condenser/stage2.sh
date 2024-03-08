#!/bin/bash

set -e

eval "$(conda shell.bash hook)"
conda activate
conda create -n condenser python=3.11
conda activate condenser

conda install pytorch=2.2.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

cd ..
pip3 install --upgrade pip
pip3 install -r ngc.txt
pip3 install -r user_experience.txt

python -m ipykernel install --user --name condenser --display-name "Python (condenser)"