#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <curand.h>
#define THREAD_NUM 256

template<typename T = dim3>
struct CudaPair 
{
  CudaPair(){};
  CudaPair(T block, T grid)
  {
    this->block = block;
    this->grid = grid;
  }
  T block, grid;
};

template<typename T = dim3>
CudaPair<T> GetCudaPair(size_t Size) 
{
  size_t NumBlocks = (Size + THREAD_NUM - 1) / THREAD_NUM;
  return CudaPair(dim3(NumBlocks, 1, 1),dim3(THREAD_NUM, 1, 1));
}

void cudaMallocInCPP(float** Input, size_t Size, size_t DeviceNum);
void DataCPUToGPU(float* CPUPointer, float* GPUPointer, size_t Size);
void DataGPUToCPU(float* CPUPointer, float* GPUPointer, size_t Size);
void DataGPUToGPU(float* GPUPointerOutput, float* GPUPointerInput, size_t Size);
void cudaFreeInCPP(float* Input);
void AddArrayInCPP(float* Output, float* InputFirst, float* InputSecond, size_t Size);
void FillArrayInCPP(float* Input, float Scalar,size_t Size);
void AddInCPP(float* Output, float* HighDimInput, size_t HighDimSize, float* LowDimInput, size_t LowDimSize);
void EleMulInCPP(float* Output, float* HighDimInput, size_t HighDimSize, float* LowDimInput, size_t LowDimSize);
void MulScalarInCPP(float* Output,float* Input, float Scalar,size_t Size);
void AddScalarInCPP(float* Output,float* Input, float Scalar,size_t Size);
void DotArrayInCPP(float* Output, float* InputFirst, float* InputSecond, size_t Size);
void MatmulInCPP
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
void TInCPP(float* Output, float* Input, size_t *MatrixShape, size_t ShapeCount);
void SumTensorDimInCPP(float* OutputData, float* InputData, size_t *InputShape, size_t InputShapeLen, size_t InputDim, size_t OutputShapeCount);
void MaximumOrMinimumTensorDimInCPP(float* OutputData, float* InputData, size_t *InputShape, size_t InputShapeLen, size_t InputDim, size_t OutputShapeCount, bool IsMaximum);
/*指定张量的任意一个维度进行拼接.*/
void TensorSpliceInCPP(float* OutputData, float* InputDataFirst, float* InputDataSecond, size_t* InputShapeFirst, size_t* InputShapeSecond, size_t InputShapeLen, size_t InputDim, size_t OutputShapeCount);
void GetUnitTensorInCPP(float* OutputData, size_t* InputShape, size_t OutputShapeCount, size_t InputShapeLen);
void GaussianEliminationInCPP(float* OutputData, size_t BatchSize, size_t Row, size_t Column);
void GetTensorBy2ShapeVectorInCPP(float* OutputData, float* InputData, size_t* InputShape,size_t* OutputShape,size_t* StartShape, size_t* EndShape, size_t ShapeLen);
void EleExpInCPP(float* OutputData, size_t OutputShape, float BaseNum);
void EleInverseInCPP(float* OutputData, size_t OutputShape);
void BroadCastToInCPP(float* OutputData, float* InputData, size_t* OutputShape, size_t* InputShape, size_t ShapeLen, size_t OutputShapeCount);
void FillRandomValNormalInCPP(float* OutputData, size_t OutputShapeCount,float MeanV, float VarianceV, unsigned Seed);
void GenerateSignTensorInCPP(float* OutputData, size_t OutputShapeCount);
void PowInCPP(float* OutputData, size_t OutputShapeCount,float Exponent);
void FillRandomValBernoulliInCPP(float* OutputData, size_t OutputShapeCount, float P, unsigned Seed);
void FillRandomValUniformInCPP(float* OutputData, size_t OutputShapeCount,float MinV, float MaxV, unsigned Seed);
void FillOnehotDataInCPP(float* OutputData, size_t BaseShape, size_t OnehotShape, size_t* InputData);
/**一堆奇怪的三角函数.*/
void TrigonometricFunctionsInCPP(float* OutputData, size_t OutputShapeCount, size_t FunType);
/**按照最后一维生成等差数列.*/
void ArithmeticSequenceInCPP(float* OutputData, size_t* OutputShape, size_t OutputShapeSize, float A1, float Arithmetic);
void GenerateTrilOnesInCPP(float* OutputData, size_t OutputShapeCount, size_t Row, size_t Col, int Diagonal);
void TransposeInCPP(float* OutputData, float* InputData, size_t* OutputShape, size_t OutputShapeSize, int FirstDim, int SecondDim);
void EleLogInCPP(float* OutputData, size_t OutputShapeSize);
void SendTensorBy2ShapeVectorInCPP(float* OutputData, float* InputData, int InputShapeCount, int* InputShapePointer, int* StartShapePointer, int* OutputShapePointer, int ShapeLen);