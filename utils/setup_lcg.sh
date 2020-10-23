#!/bin/bash

# TODO: Still still links to our local build, to be fixed as soon as ROOT 6.20 is on LCG

# Logfile to check for available GPUs
touch available_gpu.log

if uname -a | grep ekpdeepthought -q
then
    X=$(nvidia-smi | grep -n '15W' | head -1 | cut -f 1 -d ':')
    if [ $X -eq 9 ]
    then
        GPU=0
        echo "### ------ Running On GPU $GPU ------ ###"
        echo True > available_gpu.log
    elif [ $X -eq 12 ]
    then
        GPU=1
        echo "### ------ Running On GPU $GPU ------ ###"
        echo True > available_gpu.log
    elif [ $X -eq 15 ]
    then
        GPU=2
        echo "### ------ Running On GPU $GPU ------ ###"
        echo True > available_gpu.log
    elif [ $X -eq 18 ]
    then
        GPU=3
        echo "### ------ Running On GPU $GPU ------ ###"
        echo True > available_gpu.log
    else
        echo "### -- Currently No Available GPU -- ###"
        echo False > available_gpu.log
    fi

    #export CUDA_VISIBLE_DEVICES=$GPU
    export CUDA_VISIBLE_DEVICES=2
    source /home/gristo/workspace/py3_venv_ubuntu/bin/activate
    source /home/wunsch/workspace/root/build_ubuntu/bin/thisroot.sh
    export PATH=/usr/local/cuda-8.0/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda-9.0/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda-10.0/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
else
    source /cvmfs/sft.cern.ch/lcg/views/LCG_96bpython3/x86_64-centos7-gcc9-opt/setup.sh
    source /home/wunsch/workspace/root/build_own_python/bin/thisroot.sh
    alias python=/home/wunsch/workspace/python/install/bin/python3.6
    export PYTHONPATH=/home/gristo/.local/lib/python3.6/site-packages/:$PYTHONPATH
fi
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$PWD/Dumbledraw:$PYTHONPATH
