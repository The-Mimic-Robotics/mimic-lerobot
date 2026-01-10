#!/bin/bash
# Script to build PyTorch 2.7.1 from source with RTX 5070 (sm_120) support
# This will take 1-3 hours depending on your system

set -e

echo "========================================="
echo " Building PyTorch 2.7.1 with sm_120 support for RTX 5070"
echo "========================================="

# Install build dependencies
conda install -y cmake ninja mkl mkl-include

# Clone PyTorch if not already done
if [ ! -d "$HOME/pytorch" ]; then
    echo "Cloning PyTorch repository..."
    cd $HOME
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch
    git checkout v2.7.1
    git submodule sync
    git submodule update --init --recursive
else
    echo "PyTorch repository already exists"
    cd $HOME/pytorch
fi

# Set environment variables for building with sm_120
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;12.0"  # Added 12.0 for RTX 5070
export USE_CUDA=1
export USE_CUDNN=1
export USE_MKLDNN=1
export MAX_JOBS=8  # Adjust based on your CPU cores

echo ""
echo "Building PyTorch with CUDA architectures: $TORCH_CUDA_ARCH_LIST"
echo "This includes sm_120 for your RTX 5070"
echo ""
echo "Build will take 1-3 hours. Output logged to: $HOME/pytorch_build.log"
echo ""

# Clean previous build
python setup.py clean

# Build and install
python setup.py install 2>&1 | tee $HOME/pytorch_build.log

echo ""
echo "========================================="
echo " Build complete! Testing installation..."
echo "========================================="

# Test the installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA arch list: {torch.cuda.get_arch_list()}'); x=torch.randn(100,100,device='cuda'); y=x@x.T; print('✓ RTX 5070 working!')"

echo ""
echo "Success! PyTorch is now compiled with sm_120 support."
echo "You can now run your training."
