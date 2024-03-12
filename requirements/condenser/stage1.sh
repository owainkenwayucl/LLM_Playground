#!/bin/bash

set -e

# base OS setup
apt-get update
apt-get dist-upgrade -y
apt-get install --yes --no-install-recommends procps locales net-tools iputils-ping tmux screen vim htop

# set up Cuda to match host driver

# NVRM version: NVIDIA UNIX x86_64 Kernel Module  545.23.08  Mon Nov  6 23:49:37 UTC 2023
# GCC version:  gcc version 8.5.0 20210514 (Red Hat 8.5.0-21) (GCC) 

# driver_version=$(cat /proc/driver/nvidia/version | grep NVRM | awk '{print $8}')

# if [ "${driver_version}" = "550.54.14" ]
# then
#     echo "Detected Cuda 12.4 - installing into container"
#     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
#     wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2004-12-4-local_12.4.0-550.54.14-1_amd64.deb
#     mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
#     dpkg -i cuda-repo-ubuntu2004-12-4-local_12.4.0-550.54.14-1_amd64.deb
#     cp /var/cuda-repo-ubuntu2004-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
# elif [ "${driver_version}" = "545.23.08" ]
# then
#     echo "Detected Cuda 12.3 - installing into container"
#     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
#     wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda-repo-ubuntu2004-12-3-local_12.3.2-545.23.08-1_amd64.deb
#     mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
#     dpkg -i cuda-repo-ubuntu2004-12-3-local_12.3.2-545.23.08-1_amd64.deb
#     cp /var/cuda-repo-ubuntu2004-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
# elif [ "${driver_version}" = "530.30.02" ]
# then
#     echo "Detected Cuda 12.1 - installing into container"
#     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
#     wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
#     mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
#     dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
#     cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
# else
#     echo "Unknown driver installed. Exiting"
#     exit 1
# fi

# apt-get update
# apt-get dist-upgrade -y
# apt-get install -y cuda

# nvidia-smi

# initialise conda

conda init

# now we need a new shell 