#!/bin/bash

set -e

# base OS setup
apt-get update
apt-get dist-upgrade -y
apt-get install --yes --no-install-recommends procps locales net-tools iputils-ping tmux screen vim htop

# set up Cuda to match host driver
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda-repo-ubuntu2004-12-3-local_12.3.2-545.23.08-1_amd64.deb
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
dpkg -i cuda-repo-ubuntu2004-12-3-local_12.3.2-545.23.08-1_amd64.deb
cp /var/cuda-repo-ubuntu2004-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/

apt-get update
apt-get dist-upgrade -y
apt-get install -y cuda

nvidia-smi

# initialise conda

conda init

# now we need a new shell 