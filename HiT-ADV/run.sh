#!/bin/bash
# HiT-ADV launcher — handles conda/pip torch library mismatch
# Usage: bash run.sh [-m pointnet|dgcnn|pointnet++|pct] [--budget 0.55] ...

set -e

# Enable micromamba
eval "$(micromamba shell hook --shell bash 2>/dev/null)" || true
micromamba activate hitadv

# ---- resolve all library paths ----

# 1) micromamba env lib
ENV_LIB="$(dirname "$(dirname "$(python -c 'import sys; print(sys.executable)')")")/lib"

# 2) torch CUDA libs (pip-installed)
TORCH_LIB="$(python -c 'import torch; from pathlib import Path; print(Path(torch.__file__).resolve().parent / "lib")')"

# 3) system CUDA libs (AutoDL default)
CUDA_LIB="${CUDA_HOME:-/usr/local/cuda}/lib64"
[ -d "$CUDA_LIB" ] || CUDA_LIB=""

# ---- set LD_LIBRARY_PATH ----
# Order matters: torch first (for pytorch3d), then conda, then system
export LD_LIBRARY_PATH="${TORCH_LIB}:${ENV_LIB}:${CUDA_LIB:+${CUDA_LIB}:}${LD_LIBRARY_PATH}"

# Navigate to script dir
cd "$(dirname "$0")"

# Build pointnet2_ops if not yet installed
python -c "import pointnet2_ops" 2>/dev/null || pip install -e ./pointnet2_ops_lib

# Run
echo "Torch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"
echo "---"
python -u eval.py "$@"
