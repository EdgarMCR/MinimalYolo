
# Installation
In order to use the GPU, follow these steps.

Worked with Ubuntu 22.04, Python 3.11.9, Nvidia Driver Version: 535.171.04, CUDA Version: 12.2 

## CUDA + Pip Way
On Ubuntu, install CUDA 12.2 and CUDNN 8.9:
```bash
sudo apt install cuda-12-2 libcudnn8=8.9.7.29-1+cuda12.2
```
Inside the venv
```bash
pip install -r requirements.txt
```


Confirm that this works with:
```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Troubleshooting
CUDA 12.2 should be set as your chosen CUDA in Ubuntu - check this thus:
```
update-alternatives --display cuda
```

If not, use `update-alternatives --config cuda` to set it thus.
If you have path issues, check the following is in your shell's rc:
```
export PATH=/usr/local/cuda/bin:${PATH}
export CUDNN_PATH=/usr/lib/x86_64-linux-gnu/
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDNN_PATH:$CUDA_HOME/lib64:${LD_LIBRARY_PATH}
```

## Anaconda Way
** This didn't completly work on my machine but it might work for you. **

Installing 11.8 in the environment.

Inside a  conda environment ( otherwise it won't find the variables specifying the cuda and cdnn paths!)
```bash
conda install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.8.1.3 tensorflow==2.13.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
printf '#!/bin/sh\n\nunset LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

# Install NVCC
conda install -c nvidia cuda-nvcc=11.3.58

# Configure the XLA cuda directory
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
printf 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\n' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Copy libdevice file to the required path
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice
cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/
```

```bash
python3 -m pip install --upgrade tensorrt
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/tensorrt_libs:$LD_LIBRARY_PATH
```

Verify install:

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## KerasCV
Installing KerasCV can cause the following error:
```bash
AttributeError: partially initialized module 'charset_normalizer' has no attribute 'md__mypyc' (most likely due to a circular import)
```
which can be fixed using the following command:
```bash
pip install --force-reinstall charset-normalizer==3.1.0
```