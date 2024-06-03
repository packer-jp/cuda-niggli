#!/bin/sh
#------ pjsub option --------#
#PJM -g gc64

module load cuda/12.1
source ~/materials/ContinuouSP/.venv/bin/activate

cd build
cmake -DUSE_PYBIND=ON -DTorch_DIR="~/materials/ContinuouSP/.venv/lib/python3.9/site-packages/torch/share/cmake/Torch" ..
make