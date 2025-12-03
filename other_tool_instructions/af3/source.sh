#!/bin/bash
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_CLIENT_MEM_FRACTION=0.95

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PATH=$PATH:$SCRIPT_DIR/hmmer/bin