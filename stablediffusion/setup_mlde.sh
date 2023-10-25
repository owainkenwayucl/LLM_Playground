#!/bin/bash

#conda install --file requirements-conda-mlde.txt
ln -s /data/autobentham/LLM_Playground/stablediffusion/sd_generator.py /run/determined/workdir
ln -s /data/autobentham/LLM_Playground/stablediffusion/Example\ Stable\ Diffusion.ipynb /run/determined/workdir
ln -s /data/autobentham/LLM_Playground/stablediffusion/Example\ Stable\ Diffusion\ Parallel.ipynb /run/determined/workdir
ln -s /data/autobentham/LLM_Playground/stablediffusion/Image\ Viewer.ipynb /run/determined/workdir
#pip install --upgrade diffusers # we need a newer diffusers for sdxl

ln -s /data/autobentham/LLM_Playground/stablediffusion-xl/sdxl.py /run/determined/workdir
ln -s /data/autobentham/LLM_Playground/stablediffusion-xl/Stable\ Diffusion\ XL.ipynb /run/determined/workdir
ln -s /data/autobentham/LLM_Playground/stablediffusion-xl/Parallel\ SDXL.ipynb /run/determined/workdir

pip install --upgrade -r ../requirements/generative_ai_pip.txt
