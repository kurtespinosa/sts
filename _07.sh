#!/bin/bash
#$ -S /bin/bash
#$ -cwd                   # Run job from directory where submitted
#$ -V                     # Inherit environment (modulefile) settings
#$ -l k40               # Select a single GPU (Nvidia K20) node


module load apps/gcc/tensorflow/0.12.1-py27-gpu


python 07.py