#!/bin/bash

KERNEL=grouped_mm_gather_scatter_kernel

module load cuda

ncu -o profile.ncu-rep \
  -f \
  --profile-from-start no \
  --import-source yes \
  --set detailed \
  -k $KERNEL \
  python bench/profile_functional.py
