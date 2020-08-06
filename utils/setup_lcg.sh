#!/bin/bash

# TODO: Still still links to our local build, to be fixed as soon as ROOT 6.20 is on LCG

if uname -a | grep ekpdeepthought -q
then
    export CUDA_VISIBLE_DEVICES='3'
    source /home/wunsch/workspace/py3_venv_ubuntu/bin/activate
    source /home/wunsch/workspace/root/build_ubuntu/bin/thisroot.sh
    export PATH=/usr/local/cuda-8.0/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda-9.0/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda-10.0/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
    export PYTHONPATH=/home/gristo/.local/lib/python3.6/site-packages/:$PYTHONPATH
else
    source /cvmfs/sft.cern.ch/lcg/views/LCG_96bpython3/x86_64-centos7-gcc9-opt/setup.sh
    source /home/wunsch/workspace/root/build_own_python/bin/thisroot.sh
    alias python=/home/wunsch/workspace/python/install/bin/python3.6
    export PYTHONPATH=/home/gristo/.local/lib/python3.6/site-packages/:$PYTHONPATH
fi
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$PWD/Dumbledraw:$PYTHONPATH
