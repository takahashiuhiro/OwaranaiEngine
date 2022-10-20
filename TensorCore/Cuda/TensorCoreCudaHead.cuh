#include <cuda_runtime.h>
#include <stdio.h>

extern "C" void AddArray(float* Output, float* InputFirst, float* InputSecond, size_t Size);
extern "C" void cudaMallocInCPP(float** Input, size_t Size);
extern "C" void DataToGPU(float* CPUPointer, float* GPUPointer, size_t Size);
extern "C" void DataToCPU(float* CPUPointer, float* GPUPointer, size_t Size);
extern "C" void cudaFreeInCPP(float* Input);
extern "C" void AddArrayInCPP(float* Output, float* InputFirst, float* InputSecond, size_t Size);
extern "C" void FillArrayInCPP(float* Input, float Scalar,size_t Size);