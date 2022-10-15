#include "TensorCoreCudaHead.cuh"
//#include <cuda_runtime.h>

__global__ void AssignArray(float* Input, float Scalar, size_t Size) 
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if (Index < Size) Input[Index] = Scalar;
}

void CudaMallocInCpp(float* Input, size_t Size)
{
    cudaError_t err = cudaMalloc(&Input, Size*sizeof(float));
    AssignArray<<<1, Size>>>(Input, 0, Size);
}






