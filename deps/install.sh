#!/bin/bash

sudo apt-get -y install liblzma-dev libbz2-dev libzstd-dev libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev
sudo apt -y install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev
sudo apt-get -y install ffmpeg sox

pip install packaging soundfile
#cd /usr/local/lib/python3.7/dist-packages
#current_path=$PWD

new_path=/home/$USER
installation_path=$new_path/.bol/files
mkdir $installation_path

cd $installation_path

git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir -p build && cd build
cmake .. 
make -j 16
cd ..
export KENLM_ROOT=$PWD
export USE_CUDA=0 ## for cpu
cd ..


git clone https://github.com/flashlight/flashlight.git
cd flashlight/bindings/python
export USE_MKL=0
python setup.py install
