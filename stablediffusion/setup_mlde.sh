#!/bin/bash

conda install --file requirements-conda.txt
python -m ipykernel install --name=conjurer
cp /data/autobentham/LLM_Playground/stablediffusion/sd_generator.py /run/determined/workdir
cp /data/autobentham/LLM_Playground/stablediffusion/*.ipynb /run/determined/workdir