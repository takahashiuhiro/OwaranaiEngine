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
extern "C" void MatmulInCPP
(
  float* Output, 
  size_t OutputBatchShape[8], 
  size_t OutputMatrixShape[2],
  float* InputFirst, 
  size_t InputFirstBatchShape[8], 
  size_t InputFirstMatrixShape[2],
  float* InputSecond, 
  size_t InputSecondBatchShape[8], 
  size_t InputSecondMatrixShape[2],
  size_t BatchShapeLen,
  size_t OutputShapeCount,
  size_t DeviceNum
);
extern "C" void TInCPP(float* Output, float* Input, size_t *MatrixShape, size_t ShapeCount);
extern "C" void SumTensorDimInCPP(float* OutputData, float* InputData, size_t *InputShape, size_t InputShapeLen, size_t InputDim, size_t OutputShapeCount);
/*指定张量的任意一个维度进行拼接.*/
extern "C" void TensorSpliceInCPP(float* OutputData, float* InputDataFirst, float* InputDataSecond, size_t* InputShapeFirst, size_t* InputShapeSecond, size_t InputShapeLen, size_t InputDim, size_t OutputShapeCount);

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