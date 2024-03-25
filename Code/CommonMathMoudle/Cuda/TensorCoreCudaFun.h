extern "C" void cudaMallocInCPP(float** Input, size_t Size, size_t DeviceNum);
extern "C" void DataCPUToGPU(float* CPUPointer, float* GPUPointer, size_t Size);
extern "C" void DataGPUToCPU(float* CPUPointer, float* GPUPointer, size_t Size);
extern "C" void DataGPUToGPU(float* GPUPointerOutput, float* GPUPointerInput, size_t Size);
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
extern "C" void MaximumOrMinimumTensorDimInCPP(float* OutputData, float* InputData, size_t *InputShape, size_t InputShapeLen, size_t InputDim, size_t OutputShapeCount, bool IsMaximum);
/*指定张量的任意一个维度进行拼接.*/
extern "C" void TensorSpliceInCPP(float* OutputData, float* InputDataFirst, float* InputDataSecond, size_t* InputShapeFirst, size_t* InputShapeSecond, size_t InputShapeLen, size_t InputDim, size_t OutputShapeCount);
extern "C" void GetUnitTensorInCPP(float* OutputData, size_t* InputShape, size_t OutputShapeCount, size_t InputShapeLen);
extern "C" void GaussianEliminationInCPP(float* OutputData, size_t BatchSize, size_t Row, size_t Column);
extern "C" void GetTensorBy2ShapeVectorInCPP(float* OutputData, float* InputData, size_t* InputShape,size_t* OutputShape,size_t* StartShape, size_t* EndShape, size_t ShapeLen);
extern "C" void EleExpInCPP(float* OutputData, size_t OutputShape, float BaseNum);
extern "C" void EleInverseInCPP(float* OutputData, size_t OutputShape);
extern "C" void BroadCastToInCPP(float* OutputData, float* InputData, size_t* OutputShape, size_t* InputShape, size_t ShapeLen, size_t OutputShapeCount);
extern "C" void FillRandomValNormalInCPP(float* OutputData, size_t OutputShapeCount,float MeanV, float VarianceV, unsigned Seed);
extern "C" void GenerateSignTensorInCPP(float* OutputData, size_t OutputShapeCount);
extern "C" void PowInCPP(float* OutputData, size_t OutputShapeCount,float Exponent);
extern "C" void FillRandomValBernoulliInCPP(float* OutputData, size_t OutputShapeCount, float P, unsigned Seed);
extern "C" void FillRandomValUniformInCPP(float* OutputData, size_t OutputShapeCount,float MinV, float MaxV, unsigned Seed);
extern "C" void FillOnehotDataInCPP(float* OutputData, size_t BaseShape, size_t OnehotShape, size_t* InputData);
/**一堆奇怪的三角函数.*/
extern "C" void TrigonometricFunctionsInCPP(float* OutputData, size_t OutputShapeCount, size_t FunType);