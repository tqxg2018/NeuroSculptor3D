#!/usr/bin/env bash
# Create the conda environment for NeuroSculptor3D (tested: Ubuntu 22.04, NVIDIA H100/A100, CUDA 11.8 toolkit).
#
#   CUDA_HOME=/usr/local/cuda-11.8 bash install.sh            # env name: neurosculptor3d
#   ENV_NAME=myenv CUDA_HOME=/path/to/cuda-11.8 bash install.sh
#
# PyTorch 2.4.0 + CUDA 11.8 and the CUDA extensions needed by TRELLIS (xformers, spconv, kaolin, nvdiffrast,
# diffoctreerast, diff-gaussian-rasterization) plus the EMD kernel, all pinned to the versions we tested.
set -e
ENV_NAME=${ENV_NAME:-neurosculptor3d}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-11.8}
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
if ! nvcc --version | grep -q "release 11.8"; then
    echo "error: need the CUDA 11.8 toolkit (set CUDA_HOME), found: $(nvcc --version | tail -1)"; exit 1
fi
cd "$(dirname "$0")"
REPO=$(pwd)
BUILD_DIR=${BUILD_DIR:-$REPO/build/extensions}
git submodule update --init --recursive

eval "$(conda shell.bash hook)"
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"
# note: mkl>=2024.1 breaks `import torch` for the 2.4.0 conda build
conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=11.8 mkl==2023.1.0 -c pytorch -c nvidia -y

pip install -r requirements.txt
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
pip install spconv-cu118==2.3.8
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
pip install --no-deps git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

build() {  # <name> <url> <commit> <subdir to pip-install>
    if [ ! -d "$BUILD_DIR/$1" ]; then
        git clone --recurse-submodules "$2" "$BUILD_DIR/$1"
    fi
    git -C "$BUILD_DIR/$1" checkout -q "$3" && git -C "$BUILD_DIR/$1" submodule update --init --recursive
    pip install --no-build-isolation "$BUILD_DIR/$1/$4"
}
mkdir -p "$BUILD_DIR"
build nvdiffrast     https://github.com/NVlabs/nvdiffrast.git             729261dc64c4241ea36efda84fbf532cc8b425b8 .
build diffoctreerast https://github.com/JeffreyXiang/diffoctreerast.git   b09c20b84ec3aace4729e6e18a613112320eca3a .
build mip-splatting  https://github.com/autonomousvision/mip-splatting.git dda02ab5ecf45d6edb8c540d9bb65c7e451345a9 submodules/diff-gaussian-rasterization

pip install -r requirements.txt     # re-assert the pins (some of the above pull newer NumPy otherwise)

# CUDA EMD kernel for the structural metrics
pushd third_party/PyTorchEMD
python setup.py build_ext --inplace
popd

python -c "import torch, xformers, spconv, kaolin, nvdiffrast.torch, diffoctreerast, diff_gaussian_rasterization, utils3d; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"
echo "done: conda activate $ENV_NAME"
