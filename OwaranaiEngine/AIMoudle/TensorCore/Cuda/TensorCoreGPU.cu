#include "TensorCoreCudaHead.cuh"
#define THREAD_NUM 256

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

__global__ void AddArrayKernel(float* Output, float* InputFirst, float* InputSecond,size_t Size) 
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if (Index < Size) Output[Index] = InputFirst[Index] + InputSecond[Index];
}

__global__ void FillArrayKernel(float* Input, float Scalar,size_t Size) 
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if (Index < Size) Input[Index] = Scalar;
}

__global__ void AddKernel(float* Output, float* HighDimInput, size_t HighDimSize, float* LowDimInput, size_t LowDimSize) 
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if (Index < HighDimSize)Output[Index] = HighDimInput[Index] + LowDimInput[Index%LowDimSize];
}

void FillArrayInCPP(float* Input, float Scalar,size_t Size)
{
  CudaPair CudaPairInput = GetCudaPair(Size);
  FillArrayKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(Input, Scalar, Size);
}
void AddArrayInCPP(float* Output, float* InputFirst, float* InputSecond, size_t Size) 
{
  CudaPair CudaPairInput = GetCudaPair(Size);
  AddArrayKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(Output, InputFirst, InputSecond, Size);
}
void AddInCPP(float* Output, float* HighDimInput, size_t HighDimSize, float* LowDimInput, size_t LowDimSize) 
{
  CudaPair CudaPairInput = GetCudaPair(HighDimSize);
  AddKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(Output, HighDimInput, HighDimSize, LowDimInput, LowDimSize);
}

void DataToCPU(float* CPUPointer, float* GPUPointer, size_t Size){cudaMemcpy(CPUPointer,GPUPointer,sizeof(float)*Size,cudaMemcpyDeviceToHost);}
void DataToGPU(float* CPUPointer, float* GPUPointer, size_t Size){cudaMemcpy(GPUPointer,CPUPointer,sizeof(float)*Size,cudaMemcpyHostToDevice);}
void cudaFreeInCPP(float* Input){cudaFree(Input);}
void cudaMallocInCPP(float** Input, size_t Size, size_t DeviceNum)
{
  cudaSetDevice(DeviceNum);
  cudaMalloc(Input, Size*sizeof(float));
}




