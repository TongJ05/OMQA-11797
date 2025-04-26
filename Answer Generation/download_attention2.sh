#!/bin/bash
#SBATCH --job-name=install_ninja_flash
#SBATCH --output=install_%j.log
#SBATCH --error=install_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --partition=general
#SBATCH --gres=gpu:1

# Activate conda environment
echo "Activating conda environment llava_2..."
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate llava_2


# Check if ninja is installed and working correctly
echo "Checking ninja installation..."
ninja --version
NINJA_STATUS=$?

if [ $NINJA_STATUS -ne 0 ]; then
    echo "Ninja is not installed or not working correctly. Reinstalling..."
    pip uninstall -y ninja
    pip install ninja
    
    # Verify the new installation
    ninja --version
    NINJA_STATUS=$?
    
    if [ $NINJA_STATUS -ne 0 ]; then
        echo "Failed to install ninja. Exiting."
        exit 1
    else
        echo "Ninja successfully reinstalled."
    fi
else
    echo "Ninja is already installed and working correctly."
fi

# Explicitly force ninja usage
echo "Setting environment variables to force ninja usage..."
export SETUPTOOLS_USE_NINJA=1
export CMAKE_GENERATOR=Ninja
export NINJA_BUILD=1
export USE_NINJA=1
export MAX_JOBS=4

# Test if ninja is truly working
echo "Testing ninja functionality..."
mkdir -p /tmp/ninja_test
cd /tmp/ninja_test
cat > build.ninja << 'EOF'
rule cc
  command = echo "Ninja is working"

build test: cc
EOF

ninja
echo "Ninja test completed, returning to project directory."

# Install flash-attn with no-build-isolation flag
echo "Installing flash-attn with no-build-isolation flag..."
MAX_JOBS=4 pip install flash-attn==2.0.4 --no-build-isolation

# Download the LLaVA-Next model if not already downloaded
if [ ! -d "llava-next-interleave-qwen-7b" ]; then
    echo "Downloading LLaVA-Next model..."
    pip install huggingface_hub
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='lmms-lab/llava-next-interleave-qwen-7b', local_dir='/datat/user_data/ayliu2/huggingface/llava-next-interleave-qwen-7b', local_dir_use_symlinks=False)"
else
    echo "LLaVA-Next model is already downloaded."
fi

# Run the demo using the absolute path you specified
echo "Running the demo..."
python /home/ayliu2/LLaVA-NeXT/playground/demo/interleave_demo.py --model_path /datat/user_data/ayliu2/huggingface/llava-next-interleave-qwen-7b

echo "Job completed!"