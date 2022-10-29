#include "TensorCoreCudaHead.cuh"

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

__global__ void DotArrayKernel(float* Output, size_t OutSize, size_t InSize) 
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if (Index < InSize && Index + InSize < OutSize)Output[Index] += Output[Index + InSize];
  __syncthreads();
}

__global__ void AddScalarKernel(float* Output,float* Input, float Scalar,size_t Size) 
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if (Index < Size) Output[Index] = Input[Index] + Scalar;
}

__global__ void MulScalarKernel(float* Output,float* Input, float Scalar,size_t Size) 
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if (Index < Size) Output[Index] = Input[Index] * Scalar;
}

__global__ void AddKernel(float* Output, float* HighDimInput, size_t HighDimSize, float* LowDimInput, size_t LowDimSize) 
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if (Index < HighDimSize)Output[Index] = HighDimInput[Index] + LowDimInput[Index%LowDimSize];
}

__global__ void EleMulKernel(float* Output, float* HighDimInput, size_t HighDimSize, float* LowDimInput, size_t LowDimSize) 
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if (Index < HighDimSize)Output[Index] = HighDimInput[Index] * LowDimInput[Index%LowDimSize];
}

void FillArrayInCPP(float* Input, float Scalar,size_t Size)
{
  CudaPair CudaPairInput = GetCudaPair(Size);
  FillArrayKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(Input, Scalar, Size);
}

void AddScalarInCPP(float* Output,float* Input, float Scalar,size_t Size) 
{
  CudaPair CudaPairInput = GetCudaPair(Size);
  AddScalarKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(Output,Input, Scalar, Size);
}

void MulScalarInCPP(float* Output,float* Input, float Scalar,size_t Size) 
{
  CudaPair CudaPairInput = GetCudaPair(Size);
  MulScalarKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(Output,Input, Scalar, Size);
}

void AddArrayInCPP(float* Output, float* InputFirst, float* InputSecond, size_t Size) 
{
  CudaPair CudaPairInput = GetCudaPair(Size);
  AddArrayKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(Output, InputFirst, InputSecond, Size);
}

void DotArrayInCPP(float* Output, float* InputFirst, float* InputSecond, size_t Size) 
{
  CudaPair CudaPairInput = GetCudaPair(Size);
  float *OutTMP;
  cudaMalloc((void**)&OutTMP, Size*sizeof(float));
  EleMulKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutTMP, InputFirst, Size, InputSecond, Size);
  size_t SizeTMP = Size;
  while(SizeTMP > 1)
  {
    CudaPairInput = GetCudaPair(SizeTMP);
    DotArrayKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutTMP, SizeTMP, (SizeTMP + 1)/2);
    SizeTMP = (SizeTMP + 1)/2;
  }
  cudaMemcpy(Output,OutTMP,sizeof(float),cudaMemcpyDeviceToDevice);
  cudaFree(OutTMP);
}

void AddInCPP(float* Output, float* HighDimInput, size_t HighDimSize, float* LowDimInput, size_t LowDimSize) 
{
  CudaPair CudaPairInput = GetCudaPair(HighDimSize);
  AddKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(Output, HighDimInput, HighDimSize, LowDimInput, LowDimSize);
}

void EleMulInCPP(float* Output, float* HighDimInput, size_t HighDimSize, float* LowDimInput, size_t LowDimSize) 
{
  CudaPair CudaPairInput = GetCudaPair(HighDimSize);
  EleMulKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(Output, HighDimInput, HighDimSize, LowDimInput, LowDimSize);
}

void DataToCPU(float* CPUPointer, float* GPUPointer, size_t Size){cudaMemcpy(CPUPointer,GPUPointer,sizeof(float)*Size,cudaMemcpyDeviceToHost);}
void DataToGPU(float* CPUPointer, float* GPUPointer, size_t Size){cudaMemcpy(GPUPointer,CPUPointer,sizeof(float)*Size,cudaMemcpyHostToDevice);}
void cudaFreeInCPP(float* Input){cudaFree(Input);}
void cudaMallocInCPP(float** Input, size_t Size, size_t DeviceNum)
{
  cudaSetDevice(DeviceNum);
  cudaMalloc(Input, Size*sizeof(float));
}




