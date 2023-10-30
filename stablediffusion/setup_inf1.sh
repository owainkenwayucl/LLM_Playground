#!/bin/bash

python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
pip install -r ../requirements/inf1_neuron