#include "TensorCoreCudaHead.cuh"
#include <math.h>

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

__global__ void TKernel(float* Output, float* Input, size_t *MatrixShape, size_t ShapeCount)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index < ShapeCount)
  {
    size_t MatrixShapeCount = MatrixShape[0]*MatrixShape[1];
    size_t InputMatIndex = Index%MatrixShapeCount;
    size_t BaseCount = Index - InputMatIndex;
    size_t InputMatIndexFirst = InputMatIndex/MatrixShape[1];
    size_t InputMatIndexSecond = InputMatIndex%MatrixShape[1];
    Output[BaseCount + InputMatIndexSecond*MatrixShape[0] + InputMatIndexFirst] = Input[Index];
  }
}

__global__ void MatmulKernel
(
  float* Output, 
  size_t *OutputBatchShape, 
  size_t *OutputMatrixShape,
  float* InputFirst, 
  size_t *InputFirstBatchShape, 
  size_t *InputFirstMatrixShape,
  float* InputSecond, 
  size_t *InputSecondBatchShape, 
  size_t *InputSecondMatrixShape,
  size_t BatchShapeLen,
  size_t OutputShapeCount
)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if (Index < OutputShapeCount)
  {
    size_t OutputBatchIndex[8];
    size_t OutputMatrixShapeCount = OutputMatrixShape[0]*OutputMatrixShape[1];
    size_t OutSizeTMP = Index/OutputMatrixShapeCount;
    bool MatZero = OutSizeTMP;
    for(int a=BatchShapeLen-1;a>=0;a--)
    {
      if(!MatZero)OutputBatchIndex[a] = 0;
      else
      {
        OutputBatchIndex[a] = OutSizeTMP%OutputBatchShape[a];
        OutSizeTMP /= OutputBatchShape[a];
      }
    }
    size_t InputFirstBatchIndex[8];
    for(int a=BatchShapeLen-1;a>=0;a--)
    {
      if(OutputBatchIndex[a] < InputFirstBatchShape[a])InputFirstBatchIndex[a] = OutputBatchIndex[a];
      else InputFirstBatchIndex[a] = 0;
    }
    size_t InputFirstMatrixShapeCount = InputFirstMatrixShape[0]*InputFirstMatrixShape[1];
    size_t InputSecondBatchIndex[8];
    for(int a=BatchShapeLen-1;a>=0;a--)
    {
      if(OutputBatchIndex[a] < InputSecondBatchShape[a])InputSecondBatchIndex[a] = OutputBatchIndex[a];
      else InputSecondBatchIndex[a] = 0;
    }
    size_t InputSecondMatrixShapeCount = InputSecondMatrixShape[0]*InputSecondMatrixShape[1];
    size_t InputFirstBase = 0;
    size_t InFirstTMP = InputFirstMatrixShapeCount;
    for(int a=BatchShapeLen-1;a>=0;a--)
    {
      InputFirstBase += InFirstTMP*InputFirstBatchIndex[a];
      InFirstTMP*=InputFirstBatchShape[a];
    }
    size_t InputSecondBase = 0;
    size_t InSecondTMP = InputSecondMatrixShapeCount;
    for(int a=BatchShapeLen-1;a>=0;a--)
    {
      InputSecondBase += InSecondTMP*InputSecondBatchIndex[a];
      InSecondTMP*=InputSecondBatchShape[a];
    }
    size_t OutputMatrixIndex = Index%OutputMatrixShapeCount;
    size_t MatIndex[2] = {OutputMatrixIndex/OutputMatrixShape[1], OutputMatrixIndex%OutputMatrixShape[1]};
    Output[Index] = 0;
    for(int a=0;a<InputFirstMatrixShape[1];a++)
    {
      Output[Index] += InputFirst[InputFirstBase + MatIndex[0]*InputFirstMatrixShape[1] + a]*InputSecond[InputSecondBase + a*InputSecondMatrixShape[1] + MatIndex[1]];
    }
  }
}

__global__ void MaximumOrMinimumTensorDimKernel(float* OutputData, float* InputData, size_t *InputShape, size_t InputShapeLen, size_t InputDim, size_t OutputShapeCount, bool IsMaximum)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index < OutputShapeCount)
  {
    if(IsMaximum)
    {
      OutputData[Index] = -1e9+7;
    }
    else
    {
      OutputData[Index] = 1e9+7;
    }
    size_t OutputIndex[8];
    size_t OutputSizeTMP = Index;
    for(int a=InputShapeLen-1;a>=0;a--)
    {
      if(a != InputDim) 
      {
        OutputIndex[a] = OutputSizeTMP%InputShape[a];
        OutputSizeTMP /= InputShape[a];
      }
      else
      {
        OutputIndex[a] = 0;
      }
    }
    for(int a =0;a<InputShape[InputDim];a++)
    {
      size_t InputDimIndex = 0;
      size_t InputSizeTMP = 1;
      for(int b = InputShapeLen - 1;b>=0;b--)
      {
        if(b!=InputDim)InputDimIndex += InputSizeTMP*OutputIndex[b];
        else InputDimIndex += InputSizeTMP*a;
        InputSizeTMP*=InputShape[b];
      }
      if(IsMaximum)
      {
        OutputData[Index] = max(OutputData[Index], InputData[InputDimIndex]);
      }
      else
      {
        OutputData[Index] = min(OutputData[Index], InputData[InputDimIndex]);
      }
    }
  }
}

__global__ void SumTensorDimKernel(float* OutputData, float* InputData, size_t *InputShape, size_t InputShapeLen, size_t InputDim, size_t OutputShapeCount)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index < OutputShapeCount)
  {
    size_t OutputIndex[8];
    size_t OutputSizeTMP = Index;
    for(int a=InputShapeLen-1;a>=0;a--)
    {
      if(a != InputDim) 
      {
        OutputIndex[a] = OutputSizeTMP%InputShape[a];
        OutputSizeTMP /= InputShape[a];
      }
      else
      {
        OutputIndex[a] = 0;
      }
    }
    OutputData[Index] = 0;
    for(int a =0;a<InputShape[InputDim];a++)
    {
      size_t InputDimIndex = 0;
      size_t InputSizeTMP = 1;
      for(int b = InputShapeLen - 1;b>=0;b--)
      {
        if(b!=InputDim)InputDimIndex += InputSizeTMP*OutputIndex[b];
        else InputDimIndex += InputSizeTMP*a;
        InputSizeTMP*=InputShape[b];
      }
      OutputData[Index] += InputData[InputDimIndex];
    }
  }
}

__global__ void TensorSpliceKernel(float* OutputData, float* InputDataFirst, float* InputDataSecond, size_t* InputShapeFirst, size_t* InputShapeSecond, size_t InputShapeLen, size_t InputDim, size_t OutputShapeCount)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index >= OutputShapeCount)return;
  size_t RightShapeCount = 1;
  //算出指定维度右边的单元大小
  for(int a=InputDim + 1;a<InputShapeLen;a++)
  {
    RightShapeCount*= InputShapeFirst[a];
  }
  //算出指定维度的大小
  size_t InputDimCount = InputShapeFirst[InputDim] + InputShapeSecond[InputDim];
  size_t LeftDimCount = Index/RightShapeCount;
  size_t NowDimCount = LeftDimCount%InputDimCount;
  size_t StrictLeftDimCount = LeftDimCount/InputDimCount;
  if(NowDimCount < InputShapeFirst[InputDim])
  {
      OutputData[Index] = InputDataFirst[Index - StrictLeftDimCount*InputShapeSecond[InputDim]*RightShapeCount];
  }
  else
  {
      OutputData[Index] = InputDataSecond[Index - (StrictLeftDimCount+1)*InputShapeFirst[InputDim]*RightShapeCount];
  }
}

__global__ void GetUnitTensorKernel(float* OutputData, size_t* InputShape, size_t OutputShapeCount, size_t InputShapeLen)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index >= OutputShapeCount)return;
  size_t MatrixShapeCount = InputShape[InputShapeLen - 2]*InputShape[InputShapeLen - 1];
  size_t MatrixIndex = Index%MatrixShapeCount;
  if(MatrixIndex%InputShape[InputShapeLen - 2] == MatrixIndex/InputShape[InputShapeLen - 2])
  {
    OutputData[Index] = 1;
  }
}

/*找到主元.*/
__global__ void GaussianEliminationGetPivotKernel(float* OutputData, size_t BatchSize, size_t Row, size_t Column, size_t* PivotRowNumList, size_t PivotRowNum)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index >= BatchSize)return;
  for(int a=PivotRowNum;a<Row;a++)
  {
    if(OutputData[Index*Row*Column + a*Column +PivotRowNum])
    {
      PivotRowNumList[Index] = a;
      break;
    }
  }
  __syncthreads();
}

/*交换行数据.*/
__global__ void GaussianEliminationSwapRowKernel(float* OutputData, size_t BatchSize, size_t Row, size_t Column, size_t* PivotRowNumList, size_t PivotRowNum)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index >= BatchSize*Column)return;
  size_t ThisBatch = Index/Column;
  size_t ThisColumn = Index%Column;
  size_t ThisRow = PivotRowNum;
  size_t SwapRow = PivotRowNumList[ThisBatch];
  if(SwapRow != ThisRow)
  {
    float SwapTMPData = OutputData[ThisBatch*Row*Column+ThisRow*Column + ThisColumn];
    OutputData[ThisBatch*Row*Column+ThisRow*Column + ThisColumn] = OutputData[ThisBatch*Row*Column + SwapRow*Column + ThisColumn];
    OutputData[ThisBatch*Row*Column + SwapRow*Column + ThisColumn] = SwapTMPData;
  }
  __syncthreads();
}

//行除以主元的值
__global__ void GaussianEliminationNormKernel(float* OutputData, size_t BatchSize, size_t Row, size_t Column, size_t PivotRowNum)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index >= BatchSize*Column)return;
  size_t ThisBatch = Index/Column;
  size_t ThisColumn = Index%Column;
  size_t ThisRow = PivotRowNum;
  if(ThisColumn!=ThisRow)
  {
    OutputData[ThisBatch*Row*Column+ThisRow*Column + ThisColumn]/=OutputData[ThisBatch*Row*Column+ThisRow*Column + ThisRow];
  }
  __syncthreads();
}
//主元除以主元的值
__global__ void GaussianEliminationPivotNormKernel(float* OutputData, size_t BatchSize, size_t Row, size_t Column, size_t PivotRowNum)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index >= BatchSize)return;
  size_t ThisBatch = Index;
  size_t ThisColumn = PivotRowNum;
  size_t ThisRow = PivotRowNum;
  OutputData[ThisBatch*Row*Column+ThisRow*Column + ThisColumn] = 1;
  __syncthreads();
}

__global__ void GaussianEliminationMinusPivotRowKernel(float* OutputData, size_t BatchSize, size_t Row, size_t Column, size_t PivotRowNum)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index >= BatchSize*Row*Column)return;
  size_t ThisBatch = Index/(Row*Column);
  size_t ThisRow = (Index%(Row*Column)) / Column;
  size_t ThisColumn = Index%Column;
  if(ThisRow != PivotRowNum)
  {
    if(ThisColumn != PivotRowNum)
    {
      OutputData[Index] -= OutputData[ThisBatch*Row*Column + ThisRow*Column + PivotRowNum]*OutputData[ThisBatch*Row*Column + PivotRowNum*Column + ThisColumn];
    }
  }
  __syncthreads();
}

__global__ void GaussianEliminationPivotMinusPivotRowKernel(float* OutputData, size_t BatchSize, size_t Row, size_t Column, size_t PivotRowNum)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index >= BatchSize*Row)return;
  size_t ThisBatch = Index/Row;
  size_t ThisColumn = PivotRowNum;
  size_t ThisRow = Index%Row;
  if(ThisRow!=ThisColumn)
  {
    OutputData[ThisBatch*Row*Column+ThisRow*Column + ThisColumn] = 0;
  }
  __syncthreads();
}

__global__ void GetTensorBy2ShapeVectorKernel(float* OutputData, float* InputData, size_t* InputShape,size_t* OutputShape,size_t* StartShape, size_t* EndShape, size_t ShapeLen, size_t OutputShapeCount)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index >= OutputShapeCount)return;
  size_t OutputShapeIndex[10];
  size_t PreCount = Index;
  size_t InputIndex = 0;
  size_t InputIndexNowDim = 1;
  for(int a= ShapeLen -1;a>=0;a--)
  {
      OutputShapeIndex[a] =PreCount%OutputShape[a];
      OutputShapeIndex[a] += StartShape[a];
      InputIndex += OutputShapeIndex[a]*InputIndexNowDim;
      InputIndexNowDim*= InputShape[a];
      PreCount/=OutputShape[a];
  }
  OutputData[Index] = InputData[InputIndex];
}

__global__ void EleExpKernel(float* OutputData, size_t OutputShape, float BaseNum)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index >= OutputShape)return;
  OutputData[Index] = powf(BaseNum, OutputData[Index]);
}

__global__ void BroadCastToKernel(float* OutputData, float* InputData, size_t* OutputShape, size_t* InputShape, size_t ShapeLen, size_t OutputShapeCount)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index >= OutputShapeCount)return;
  size_t ShapeIndex[10];
  size_t NowIndex = Index;
  for(int a = ShapeLen - 1 ;a >= 0;a--)
  {
    ShapeIndex[a] = NowIndex%OutputShape[a];
    NowIndex = size_t(NowIndex/OutputShape[a]);
    if(OutputShape[a] > InputShape[a])ShapeIndex[a] = 0;
  }
  size_t FixedInputIndex = 0;
  for(size_t a = 0;a<ShapeLen;a++)
  {
    FixedInputIndex *= InputShape[a];
    FixedInputIndex += ShapeIndex[a];
  }
  OutputData[Index] = InputData[FixedInputIndex];
}

__global__ void EleInverseKernel(float* OutputData, size_t OutputShape)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index >= OutputShape)return;
  OutputData[Index] = 1./OutputData[Index];
}

__global__ void GenerateSignTensorKernel(float* OutputData, size_t OutputShapeCount, float SwitchValue)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index >= OutputShapeCount)return;
  if(OutputData[Index] > SwitchValue)OutputData[Index] = 1.;
  else OutputData[Index] = 0.;
}

__global__ void PowKernel(float* OutputData, size_t OutputShapeCount,float Exponent)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index >= OutputShapeCount)return;
  OutputData[Index] = pow(OutputData[Index], Exponent);
}

__global__ void FillOnehotDataKernel(float* OutputData, size_t BaseShape, size_t OnehotShape, size_t* InputData)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index >= BaseShape)return;
  OutputData[Index*OnehotShape+InputData[Index]] = 1.;
}

__global__ void TrigonometricFunctionsKernel(float* OutputData, size_t OutputShapeCount, size_t FunType)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index >= OutputShapeCount)return;
  if(FunType == 0)OutputData[Index] = sin(OutputData[Index]);
  if(FunType == 1)OutputData[Index] = cos(OutputData[Index]);
}

__global__ void ArithmeticSequenceKernel(float* OutputData, size_t* OutputShape, size_t OutputShapeSize, float A1, float Arithmetic)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t OutputShapeCount = 1;
  for(size_t a=0;a<OutputShapeSize;a++)OutputShapeCount *= OutputShape[a];
  if(Index >= OutputShapeCount)return;
  size_t CurIndex = Index%OutputShape[OutputShapeSize-1];
  OutputData[Index] = CurIndex*Arithmetic + A1;
}

__global__ void GenerateTrilOnesKernel(float* OutputData,size_t OutputShapeCount, size_t Row, size_t Col, int Diagonal)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index >= OutputShapeCount)return;
  size_t TrueIndex = Index%(Row*Col);
  int ThisRow = TrueIndex/Col;
  int ThisCol = TrueIndex%Col;
  if(ThisCol <= ThisRow + Diagonal)OutputData[Index] = 1;
  else OutputData[Index] = 0;
}

__global__ void TransposeKernel(float* OutputData, float* InputData, size_t* OutputShape, size_t OutputShapeSize, int FirstDim, int SecondDim)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t OutputShapeCount = 1;
  int LeftDim = min(FirstDim, SecondDim);//最左边的维度
  int RightDim = max(FirstDim, SecondDim);//最右边的维度
  int MidRightElement = 1;//所有的要变换的维度的元素个数
  int RightElement = 1;//比要变换的右边的维度还小的
  int MidElement = 1;//中间的维度
  for(size_t a=0;a<OutputShapeSize;a++)
  {
    OutputShapeCount*=OutputShape[a];
    if(a < LeftDim)continue;
    MidRightElement*=OutputShape[a];
    if(a > RightDim)RightElement*=OutputShape[a];
    if(a < RightDim&&a > LeftDim)MidElement*=OutputShape[a];
  }
  if(Index >= OutputShapeCount)return;
  //交换维度得到输入的shape
  int NowIndex = Index%MidRightElement;
  int UseIndex = NowIndex/RightElement;//右边都是行向量
  int ReduIndex = NowIndex%RightElement;
  int ADim = UseIndex/(MidElement*OutputShape[RightDim]);
  int BDim = (UseIndex%(MidElement*OutputShape[RightDim]))/OutputShape[RightDim];
  int CDim = UseIndex%OutputShape[RightDim];
  int InputUseIndex = CDim*(MidElement*OutputShape[LeftDim]) + BDim*OutputShape[LeftDim] + ADim;
  int InputNowIndex = InputUseIndex*RightElement + ReduIndex;
  int InputIndex = InputNowIndex+(int(Index/MidRightElement))*MidRightElement;
  OutputData[Index] = InputData[InputIndex];
}

__global__ void EleLogKernel(float* OutputData, size_t OutputShapeSize)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index >= OutputShapeSize)return;
  OutputData[Index] = log(OutputData[Index]);
}

__global__ void SendTensorBy2ShapeVectorKernel(float* OutputData, float* InputData, int InputShapeCount, int* InputShapePointer, int* StartShapePointer, int* OutputShapePointer, int ShapeLen)
{
  size_t Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index >= InputShapeCount)return;
  int InputShapeIndex[10];
  int PreIndex = int(Index);
  for(int a=ShapeLen-1;a>=0;a--)
  {
      int ThisShapeIndex = a;
      InputShapeIndex[ThisShapeIndex] = PreIndex%InputShapePointer[ThisShapeIndex];
      PreIndex = PreIndex/InputShapePointer[ThisShapeIndex];
  }
  int OutputShapeIndex[10];
  for(int a=ShapeLen-1;a>=0;a--)OutputShapeIndex[a] = InputShapeIndex[a] + StartShapePointer[a];
  int OutputIndex = 0;
  for(int a=0;a<ShapeLen;a++)
  {
      OutputIndex *= OutputShapePointer[a];
      OutputIndex += OutputShapeIndex[a];
  }
  OutputData[OutputIndex] = InputData[Index];
}

void SendTensorBy2ShapeVectorInCPP(float* OutputData, float* InputData, int InputShapeCount, int* InputShapePointer, int* StartShapePointer, int* OutputShapePointer, int ShapeLen)
{
  CudaPair CudaPairInput = GetCudaPair(InputShapeCount);
  int *InputShapePointerCuda;
  cudaMalloc((void**)&InputShapePointerCuda, ShapeLen*sizeof(int));
  cudaMemcpy(InputShapePointerCuda,InputShapePointer,ShapeLen*sizeof(int),cudaMemcpyHostToDevice);
  int *StartShapePointerCuda;
  cudaMalloc((void**)&StartShapePointerCuda, ShapeLen*sizeof(int));
  cudaMemcpy(StartShapePointerCuda,StartShapePointer,ShapeLen*sizeof(int),cudaMemcpyHostToDevice);
  int *OutputShapePointerCuda;
  cudaMalloc((void**)&OutputShapePointerCuda, ShapeLen*sizeof(int));
  cudaMemcpy(OutputShapePointerCuda,OutputShapePointer,ShapeLen*sizeof(int),cudaMemcpyHostToDevice);
  SendTensorBy2ShapeVectorKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutputData, InputData, InputShapeCount, InputShapePointerCuda, StartShapePointerCuda, OutputShapePointerCuda, ShapeLen);
  cudaFree(InputShapePointerCuda);
  cudaFree(StartShapePointerCuda);
  cudaFree(OutputShapePointerCuda);
}

void EleLogInCPP(float* OutputData, size_t OutputShapeSize)
{
  CudaPair CudaPairInput = GetCudaPair(OutputShapeSize);
  EleLogKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutputData, OutputShapeSize);
}

void TransposeInCPP(float* OutputData, float* InputData, size_t* OutputShape, size_t OutputShapeSize, int FirstDim, int SecondDim)
{
  size_t OutputShapeCount = 1;
  for(size_t a=0;a<OutputShapeSize;a++)OutputShapeCount*=OutputShape[a];
  CudaPair CudaPairInput = GetCudaPair(OutputShapeCount);
  size_t *OutputShapeCuda;
  cudaMalloc((void**)&OutputShapeCuda, OutputShapeSize*sizeof(size_t));
  cudaMemcpy(OutputShapeCuda,OutputShape,sizeof(size_t)*OutputShapeSize,cudaMemcpyHostToDevice);
  TransposeKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutputData, InputData, OutputShapeCuda, OutputShapeSize, FirstDim, SecondDim);
  cudaFree(OutputShapeCuda);
}

void GenerateTrilOnesInCPP(float* OutputData,size_t OutputShapeCount, size_t Row, size_t Col, int Diagonal)
{
  CudaPair CudaPairInput = GetCudaPair(OutputShapeCount);
  GenerateTrilOnesKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutputData, OutputShapeCount, Row, Col, Diagonal);
}

void ArithmeticSequenceInCPP(float* OutputData, size_t* OutputShape, size_t OutputShapeSize, float A1, float Arithmetic)
{
  size_t OutputShapeCount = 1;
  for(size_t a=0;a<OutputShapeSize;a++)OutputShapeCount *= OutputShape[a];
  CudaPair CudaPairInput = GetCudaPair(OutputShapeCount);
  size_t *OutputShapeCuda;
  cudaMalloc((void**)&OutputShapeCuda, OutputShapeSize*sizeof(size_t));
  cudaMemcpy(OutputShapeCuda,OutputShape,sizeof(size_t)*OutputShapeSize,cudaMemcpyHostToDevice);
  ArithmeticSequenceKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutputData, OutputShapeCuda, OutputShapeSize, A1, Arithmetic);
  cudaFree(OutputShapeCuda);
}

void TrigonometricFunctionsInCPP(float* OutputData, size_t OutputShapeCount, size_t FunType)
{
  CudaPair CudaPairInput = GetCudaPair(OutputShapeCount);
  TrigonometricFunctionsKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutputData, OutputShapeCount,FunType);
}

void FillOnehotDataInCPP(float* OutputData, size_t BaseShape, size_t OnehotShape, size_t* InputData)
{
  CudaPair CudaPairInput = GetCudaPair(BaseShape);
  size_t *InputDataCuda;
  cudaMalloc((void**)&InputDataCuda, BaseShape*sizeof(size_t));
  cudaMemcpy(InputDataCuda,InputData,sizeof(size_t)*BaseShape,cudaMemcpyHostToDevice);
  FillOnehotDataKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutputData, BaseShape, OnehotShape, InputDataCuda);
  cudaFree(InputDataCuda);
}

void PowInCPP(float* OutputData, size_t OutputShapeCount,float Exponent)
{
  CudaPair CudaPairInput = GetCudaPair(OutputShapeCount);
  PowKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutputData, OutputShapeCount,Exponent);
}

void GenerateSignTensorInCPP(float* OutputData, size_t OutputShapeCount)
{
  CudaPair CudaPairInput = GetCudaPair(OutputShapeCount);
  GenerateSignTensorKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutputData, OutputShapeCount, 0);
}

void EleInverseInCPP(float* OutputData, size_t OutputShape)
{
  CudaPair CudaPairInput = GetCudaPair(OutputShape);
  EleInverseKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutputData, OutputShape);
}

void BroadCastToInCPP(float* OutputData, float* InputData, size_t* OutputShape, size_t* InputShape, size_t ShapeLen, size_t OutputShapeCount)
{
  size_t *InputShapeCuda;
  size_t *OutputShapeCuda;
  cudaMalloc((void**)&InputShapeCuda, ShapeLen*sizeof(size_t));
  cudaMalloc((void**)&OutputShapeCuda, ShapeLen*sizeof(size_t));
  cudaMemcpy(InputShapeCuda,InputShape,sizeof(size_t)*ShapeLen,cudaMemcpyHostToDevice);
  cudaMemcpy(OutputShapeCuda,OutputShape,sizeof(size_t)*ShapeLen,cudaMemcpyHostToDevice);
  CudaPair CudaPairInput = GetCudaPair(OutputShapeCount);
  BroadCastToKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutputData, InputData, OutputShapeCuda, InputShapeCuda, ShapeLen, OutputShapeCount);
  cudaFree(InputShapeCuda);
  cudaFree(OutputShapeCuda);
}

void EleExpInCPP(float* OutputData, size_t OutputShape, float BaseNum)
{
  CudaPair CudaPairInput = GetCudaPair(OutputShape);
  EleExpKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutputData, OutputShape, BaseNum);
}

void GetTensorBy2ShapeVectorInCPP(float* OutputData, float* InputData, size_t* InputShape,size_t* OutputShape,size_t* StartShape, size_t* EndShape, size_t ShapeLen)
{
  size_t *InputShapeCuda;
  size_t *OutputShapeCuda;
  size_t *StartShapeCuda;
  size_t *EndShapeCuda;
  cudaMalloc((void**)&InputShapeCuda, ShapeLen*sizeof(size_t));
  cudaMalloc((void**)&OutputShapeCuda, ShapeLen*sizeof(size_t));
  cudaMalloc((void**)&StartShapeCuda, ShapeLen*sizeof(size_t));
  cudaMalloc((void**)&EndShapeCuda, ShapeLen*sizeof(size_t));
  cudaMemcpy(InputShapeCuda,InputShape,sizeof(size_t)*ShapeLen,cudaMemcpyHostToDevice);
  cudaMemcpy(OutputShapeCuda,OutputShape,sizeof(size_t)*ShapeLen,cudaMemcpyHostToDevice);
  cudaMemcpy(StartShapeCuda,StartShape,sizeof(size_t)*ShapeLen,cudaMemcpyHostToDevice);
  cudaMemcpy(EndShapeCuda,EndShape,sizeof(size_t)*ShapeLen,cudaMemcpyHostToDevice);
  size_t OutputShapeCount = 1;
  for(int a=0;a<ShapeLen;a++)
  {
    OutputShapeCount*= OutputShape[a];
  }
  CudaPair CudaPairInput = GetCudaPair(OutputShapeCount);
  GetTensorBy2ShapeVectorKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutputData, InputData, InputShapeCuda,OutputShapeCuda,StartShapeCuda,EndShapeCuda,ShapeLen,OutputShapeCount);
  cudaFree(InputShapeCuda);
  cudaFree(OutputShapeCuda);
  cudaFree(StartShapeCuda);
  cudaFree(EndShapeCuda);
}

void GaussianEliminationInCPP(float* OutputData, size_t BatchSize, size_t Row, size_t Column)
{
  CudaPair CudaPairInputRow = GetCudaPair(BatchSize*Column);
  CudaPair CudaPairInputBatch = GetCudaPair(BatchSize);
  CudaPair CudaPairInputAllData = GetCudaPair(BatchSize*Column*Row);
  CudaPair CudaPairInputColumn = GetCudaPair(BatchSize*Row);
  size_t* PivotRowNumList;
  cudaMalloc((void**)&PivotRowNumList, BatchSize*sizeof(size_t));
  for(int a=0;a<Row;a++)
  {
    GaussianEliminationGetPivotKernel<<<CudaPairInputBatch.block, CudaPairInputBatch.grid>>>(OutputData, BatchSize, Row, Column, PivotRowNumList, a);
    GaussianEliminationSwapRowKernel<<<CudaPairInputRow.block, CudaPairInputRow.grid>>>(OutputData,BatchSize, Row, Column, PivotRowNumList, a);
    GaussianEliminationNormKernel<<<CudaPairInputRow.block, CudaPairInputRow.grid>>>(OutputData,BatchSize, Row, Column,a);
    GaussianEliminationPivotNormKernel<<<CudaPairInputBatch.block, CudaPairInputBatch.grid>>>(OutputData,BatchSize, Row, Column,a);
    GaussianEliminationMinusPivotRowKernel<<<CudaPairInputAllData.block, CudaPairInputAllData.grid>>>(OutputData,BatchSize, Row, Column,a);
    GaussianEliminationPivotMinusPivotRowKernel<<<CudaPairInputColumn.block, CudaPairInputColumn.grid>>>(OutputData,BatchSize, Row, Column,a);
  }
}

void GetUnitTensorInCPP(float* OutputData, size_t* InputShape, size_t OutputShapeCount, size_t InputShapeLen)
{
  size_t *InputShapeCuda;
  cudaMalloc((void**)&InputShapeCuda, InputShapeLen*sizeof(size_t));
  cudaMemcpy(InputShapeCuda,InputShape,sizeof(size_t)*InputShapeLen,cudaMemcpyHostToDevice);
  CudaPair CudaPairInput = GetCudaPair(OutputShapeCount);
  GetUnitTensorKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutputData, InputShapeCuda, OutputShapeCount, InputShapeLen);
  cudaFree(InputShapeCuda);
}

void TensorSpliceInCPP(float* OutputData, float* InputDataFirst, float* InputDataSecond, size_t* InputShapeFirst, size_t* InputShapeSecond, size_t InputShapeLen, size_t InputDim, size_t OutputShapeCount)
{
  size_t *InputShapeFirstCuda;
  size_t *InputShapeSecondCuda;
  cudaMalloc((void**)&InputShapeFirstCuda, InputShapeLen*sizeof(size_t));
  cudaMalloc((void**)&InputShapeSecondCuda, InputShapeLen*sizeof(size_t));
  cudaMemcpy(InputShapeFirstCuda,InputShapeFirst,sizeof(size_t)*InputShapeLen,cudaMemcpyHostToDevice);
  cudaMemcpy(InputShapeSecondCuda,InputShapeSecond,sizeof(size_t)*InputShapeLen,cudaMemcpyHostToDevice);
  CudaPair CudaPairInput = GetCudaPair(OutputShapeCount);
  TensorSpliceKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutputData, InputDataFirst, InputDataSecond, InputShapeFirstCuda, InputShapeSecondCuda, InputShapeLen, InputDim, OutputShapeCount);
  cudaFree(InputShapeFirstCuda);
  cudaFree(InputShapeSecondCuda);
}

void SumTensorDimInCPP(float* OutputData, float* InputData, size_t *InputShape, size_t InputShapeLen, size_t InputDim, size_t OutputShapeCount)
{
  size_t *InputShapeCuda;
  cudaMalloc((void**)&InputShapeCuda, InputShapeLen*sizeof(size_t));
  cudaMemcpy(InputShapeCuda,InputShape,sizeof(size_t)*InputShapeLen,cudaMemcpyHostToDevice);
  CudaPair CudaPairInput = GetCudaPair(OutputShapeCount);
  SumTensorDimKernel<<<CudaPairInput.block, CudaPairInput.grid>>>( OutputData, InputData, InputShapeCuda, InputShapeLen, InputDim, OutputShapeCount);
  cudaFree(InputShapeCuda);
}

void MaximumOrMinimumTensorDimInCPP(float* OutputData, float* InputData, size_t *InputShape, size_t InputShapeLen, size_t InputDim, size_t OutputShapeCount, bool IsMaximum)
{
  size_t *InputShapeCuda;
  cudaMalloc((void**)&InputShapeCuda, InputShapeLen*sizeof(size_t));
  cudaMemcpy(InputShapeCuda,InputShape,sizeof(size_t)*InputShapeLen,cudaMemcpyHostToDevice);
  CudaPair CudaPairInput = GetCudaPair(OutputShapeCount);
  MaximumOrMinimumTensorDimKernel<<<CudaPairInput.block, CudaPairInput.grid>>>( OutputData, InputData, InputShapeCuda, InputShapeLen, InputDim, OutputShapeCount, IsMaximum);
  cudaFree(InputShapeCuda);
}


void TInCPP(float* Output, float* Input, size_t *MatrixShape, size_t ShapeCount)
{
  size_t *MatrixShapeCuda;
  cudaMalloc((void**)&MatrixShapeCuda, 2*sizeof(size_t));
  cudaMemcpy(MatrixShapeCuda,MatrixShape,sizeof(size_t)*2,cudaMemcpyHostToDevice);
  CudaPair CudaPairInput = GetCudaPair(ShapeCount);
  TKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(Output, Input, MatrixShapeCuda, ShapeCount);
  cudaFree(MatrixShapeCuda);
}

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
)
{
  cudaSetDevice(DeviceNum);
  size_t *OutputBatchShapeCuda;
  cudaMalloc((void**)&OutputBatchShapeCuda, 8*sizeof(size_t));
  cudaMemcpy(OutputBatchShapeCuda,OutputBatchShape,sizeof(size_t)*8,cudaMemcpyHostToDevice);
  size_t *OutputMatrixShapeCuda;
  cudaMalloc((void**)&OutputMatrixShapeCuda, 2*sizeof(size_t));
  cudaMemcpy(OutputMatrixShapeCuda,OutputMatrixShape,sizeof(size_t)*2,cudaMemcpyHostToDevice);
  size_t *InputFirstBatchShapeCuda;
  cudaMalloc((void**)&InputFirstBatchShapeCuda, 8*sizeof(size_t));
  cudaMemcpy(InputFirstBatchShapeCuda,InputFirstBatchShape,sizeof(size_t)*8,cudaMemcpyHostToDevice);
  size_t *InputFirstMatrixShapeCuda;
  cudaMalloc((void**)&InputFirstMatrixShapeCuda, 2*sizeof(size_t));
  cudaMemcpy(InputFirstMatrixShapeCuda,InputFirstMatrixShape,sizeof(size_t)*2,cudaMemcpyHostToDevice);
  size_t *InputSecondBatchShapeCuda;
  cudaMalloc((void**)&InputSecondBatchShapeCuda, 8*sizeof(size_t));
  cudaMemcpy(InputSecondBatchShapeCuda,InputSecondBatchShape,sizeof(size_t)*8,cudaMemcpyHostToDevice);
  size_t *InputSecondMatrixShapeCuda;
  cudaMalloc((void**)&InputSecondMatrixShapeCuda, 2*sizeof(size_t));
  cudaMemcpy(InputSecondMatrixShapeCuda,InputSecondMatrixShape,sizeof(size_t)*2,cudaMemcpyHostToDevice);
  CudaPair CudaPairInput = GetCudaPair(OutputShapeCount);
  MatmulKernel<<<CudaPairInput.block, CudaPairInput.grid>>>
  (
    Output, 
    OutputBatchShapeCuda, 
    OutputMatrixShapeCuda, 
    InputFirst,
    InputFirstBatchShapeCuda, 
    InputFirstMatrixShapeCuda,
    InputSecond, 
    InputSecondBatchShapeCuda,
    InputSecondMatrixShapeCuda,
    BatchShapeLen,
    OutputShapeCount
  );
  cudaFree(OutputBatchShapeCuda);
  cudaFree(OutputMatrixShapeCuda);
  cudaFree(InputFirstBatchShapeCuda);
  cudaFree(InputFirstMatrixShapeCuda);
  cudaFree(InputSecondBatchShapeCuda);
  cudaFree(InputSecondMatrixShapeCuda);
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

void DataGPUToCPU(float* CPUPointer, float* GPUPointer, size_t Size){cudaMemcpy(CPUPointer,GPUPointer,sizeof(float)*Size,cudaMemcpyDeviceToHost);}
void DataCPUToGPU(float* CPUPointer, float* GPUPointer, size_t Size){cudaMemcpy(GPUPointer,CPUPointer,sizeof(float)*Size,cudaMemcpyHostToDevice);}
void DataGPUToGPU(float* GPUPointerOutput, float* GPUPointerInput, size_t Size){cudaMemcpy(GPUPointerOutput,GPUPointerInput,sizeof(float)*Size,cudaMemcpyDeviceToDevice);}
void cudaFreeInCPP(float* Input){cudaFree(Input);}
void cudaMallocInCPP(float** Input, size_t Size, size_t DeviceNum)
{
  cudaSetDevice(DeviceNum);
  cudaMalloc(Input, Size*sizeof(float));
}

void FillRandomValNormalInCPP(float* OutputData, size_t OutputShapeCount,float MeanV, float VarianceV, unsigned Seed)
{
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, Seed);
  curandGenerateNormal(gen, OutputData, OutputShapeCount+OutputShapeCount%2, MeanV, VarianceV);
}

void FillRandomValBernoulliInCPP(float* OutputData, size_t OutputShapeCount, float P, unsigned Seed)
{
  FillRandomValUniformInCPP(OutputData, OutputShapeCount,0,1, Seed);
  CudaPair CudaPairInput = GetCudaPair(OutputShapeCount);
  GenerateSignTensorKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutputData, OutputShapeCount, P);
}

void FillRandomValUniformInCPP(float* OutputData, size_t OutputShapeCount,float MinV, float MaxV, unsigned Seed)
{
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, Seed);
  curandGenerateUniform(gen, OutputData, OutputShapeCount);
  if(MinV*MinV < 1e-7 && (MaxV*MaxV-2*MaxV+1)<1e-7)return;
  CudaPair CudaPairInput = GetCudaPair(OutputShapeCount);
  MulScalarKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutputData,OutputData, MaxV-MinV, OutputShapeCount);
  AddScalarKernel<<<CudaPairInput.block, CudaPairInput.grid>>>(OutputData,OutputData, MinV, OutputShapeCount);
}