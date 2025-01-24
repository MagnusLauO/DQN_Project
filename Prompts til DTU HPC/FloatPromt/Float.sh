#!/bin/sh
#BSUB -J Float
#BSUB -o Float%J.out
#BSUB -e Float%J.err
#BSUB -n 4
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 08:00
# end of BSUB options

export XDG_RUNTIME_DIR=/tmp/$USER-runtime
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

module load cuda/12.4

python3 DQNBallsFloat.py