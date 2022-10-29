#include <cuda_runtime.h>
#include <stdio.h>
#define THREAD_NUM 256

extern "C" void cudaMallocInCPP(float** Input, size_t Size, size_t DeviceNum);
extern "C" void DataToGPU(float* CPUPointer, float* GPUPointer, size_t Size);
extern "C" void DataToCPU(float* CPUPointer, float* GPUPointer, size_t Size);
extern "C" void cudaFreeInCPP(float* Input);
extern "C" void AddArrayInCPP(float* Output, float* InputFirst, float* InputSecond, size_t Size);
extern "C" void FillArrayInCPP(float* Input, float Scalar,size_t Size);
extern "C" void AddInCPP(float* Output, float* HighDimInput, size_t HighDimSize, float* LowDimInput, size_t LowDimSize);
extern "C" void EleMulInCPP(float* Output, float* HighDimInput, size_t HighDimSize, float* LowDimInput, size_t LowDimSize);
extern "C" void MulScalarInCPP(float* Output,float* Input, float Scalar,size_t Size);
extern "C" void AddScalarInCPP(float* Output,float* Input, float Scalar,size_t Size);
extern "C" void DotArrayInCPP(float* Output, float* InputFirst, float* InputSecond, size_t Size);

struct CudaPair 
{
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
  return CudaPair(dim3(THREAD_NUM, 1, 1), dim3(NumBlocks, 1, 1));
}