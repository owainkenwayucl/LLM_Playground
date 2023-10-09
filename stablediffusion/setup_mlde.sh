#!/bin/bash

conda install --file requirements-conda-mlde.txt
python -m ipykernel install --name=conjurer
ln -s /data/autobentham/LLM_Playground/stablediffusion/sd_generator.py /run/determined/workdir
ln -s /data/autobentham/LLM_Playground/stablediffusion/Example\ Stable\ Diffusion.ipynb /run/determined/workdir
ln -s /data/autobentham/LLM_Playground/stablediffusion/Image\ Viewer.ipynb /run/determined/workdir
