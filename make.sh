#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda

echo "Compiling roi pooling kernels by nvcc..."
cd lib/layer_utils/roi_pooling/src/cuda
$CUDA_PATH/bin/nvcc -c -o roi_pooling.cu.o roi_pooling_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../../
python build.py


cd ../psroi_pooling/src/cuda
echo "Compiling psroi pooling kernels by nvcc..."

$CUDA_PATH/bin/nvcc -c -o psroi_pooling.cu.o psroi_pooling_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../../
python build.py


cd ../../nms/src/cuda
echo "Compiling nms kernels by nvcc..."
$CUDA_PATH/bin/nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ../../
python build.py