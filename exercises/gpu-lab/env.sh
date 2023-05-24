#!/bin/bash

module load 2022r2
module load intel/oneapi-all
module load likwid

if [[ "${HOSTNAME}" =~ "gpu" ]]; then
  module load nvhpc
  module load cuda/11.1.1
fi
