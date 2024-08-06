#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <curand.h>
#include "TensorCoreCudaFun.h"
#define THREAD_NUM 256

struct CudaPair 
{
  CudaPair(){};
  CudaPair(dim3 block, dim3 grid)
  {
    this->block = block;
    this->grid = grid;
  }
  dim3 block, grid;
};

CudaPair GetCudaPair(size_t Size) 
{
  size_t NumBlocks = (Size + THREAD_NUM - 1) / THREAD_NUM;
  return CudaPair(dim3(NumBlocks, 1, 1),dim3(THREAD_NUM, 1, 1));
}