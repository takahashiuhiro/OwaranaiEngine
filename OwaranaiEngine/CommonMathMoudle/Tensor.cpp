#include "Tensor.h"

Tensor::Tensor(std::vector<size_t>shape)
{
    this->shape = shape;
    for(int a=0;a<shape.size();a++)ShapeCount*=shape[a];
    this->DataCPU = (float*)malloc(sizeof(float)*ShapeCount);
}

Tensor::Tensor(std::vector<size_t>shape, std::string Device, size_t DeviceNum)
{
    this->shape = shape;
    this->Device = Device;
    this->DeviceNum = DeviceNum;
    for(int a=0;a<shape.size();a++)ShapeCount*=shape[a];
    if(Device == "GPU")
    {
        #ifdef CUDA_USEFUL
        cudaMallocInCPP(&(this->DataGPU), ShapeCount, DeviceNum);
        #endif
    }
    else this->DataCPU = (float*)malloc(sizeof(float)*ShapeCount);
}

std::vector<size_t> Tensor::GetDim(size_t DataIndex)
{
    std::vector<size_t>ReturnVector;
    for(int a=0;a<shape.size();a++)ReturnVector.push_back(0);
    for(int a=shape.size()-1;a>=0;a--)
    {
        ReturnVector[a] = DataIndex%(shape[a]);
        DataIndex /= shape[a];
    }
    return ReturnVector;
}

void Tensor::PrintData()
{
    bool GPUflag = 0;
    if(Device == "GPU")
    {
        #ifdef CUDA_USEFUL
        GPUflag = 1;
        ToCPU();
        #endif
    }
    for(int a=0;a<ShapeCount;a++)
    {
        std::cout<< std::fixed << std::setprecision(3) <<DataCPU[a]<<" ";
        std::vector<size_t>DimIndex = GetDim(a);
        for(int b=shape.size() - 1;b>=0;b--)
        {
            if(DimIndex[b] == shape[b] - 1)
            {
                if(b == shape.size() - 2)
                {
                    std::cout<<"DimIndex:(";
                    for(int c =0 ; c<shape.size()-2;c++)std::cout<<DimIndex[c]<<",";
                    std::cout<<")\n";
                }
                if(b > shape.size() - 2)std::cout<<std::endl;
            }
            else break;
        }
    }
    std::cout<<std::endl;
    if(GPUflag)ToGPU();
}
void Tensor::ToCPU()
{
    #ifdef CUDA_USEFUL
    if(Device == "CPU")return;
    this->DataCPU = (float*)malloc(sizeof(float)*ShapeCount);
    DataToCPU(DataCPU, DataGPU, ShapeCount);
    cudaFreeInCPP(DataGPU);
    Device = "CPU";
    #endif
}

void Tensor::ToGPU()
{
    #ifdef CUDA_USEFUL
    if(Device == "GPU")return;
    cudaMallocInCPP(&DataGPU, ShapeCount, DeviceNum);
    DataToGPU(DataCPU, DataGPU, ShapeCount);
    free(DataCPU);
    Device = "GPU";
    #endif
}

void Tensor::FillArray(float Scalar)
{
    if(Device == "GPU")
    {
        #ifdef CUDA_USEFUL
        FillArrayInCPP(DataGPU, Scalar, ShapeCount);
        #endif
    }
    else for(int a=0;a<ShapeCount;a++)DataCPU[a] = Scalar;
}

Tensor* Tensor::AddScalar(float Scalar)
{
    Tensor* Output = new Tensor(shape, Device, DeviceNum);
    if(Device == "GPU")
    {
        #ifdef CUDA_USEFUL
        AddScalarInCPP(Output->DataGPU, DataGPU, Scalar, ShapeCount);
        #endif
    }
    else for(int a=0;a<ShapeCount;a++)Output->DataCPU[a] = DataCPU[a]+ Scalar;
    return Output;
}

Tensor* Tensor::MulScalar(float Scalar)
{
    Tensor* Output = new Tensor(shape, Device, DeviceNum);
    if(Device == "GPU")
    {
        #ifdef CUDA_USEFUL
        MulScalarInCPP(Output->DataGPU, DataGPU, Scalar, ShapeCount);
        #endif
    }
    else for(int a=0;a<ShapeCount;a++)Output->DataCPU[a] = DataCPU[a] *Scalar;
    return Output;
}

Tensor* Tensor::AddArray(Tensor* Input)
{
    Tensor* Output = new Tensor(shape, Device, DeviceNum);
    if(Device == "GPU")
    {
        #ifdef CUDA_USEFUL
        AddArrayInCPP(Output->DataGPU, DataGPU, Input->DataGPU, ShapeCount);
        #endif
    }
    else for(int a=0;a<ShapeCount;a++)Output->DataCPU[a] = DataCPU[a] + Input->DataCPU[a];
    return Output;
}

Tensor* Tensor::Add(Tensor* Input)
{
    Tensor *Output, *HighDimTensor, *LowDimTensor;
    if(shape.size() > Input->shape.size())
    {
        HighDimTensor = this;
        LowDimTensor = Input;
    }
    else
    {
        HighDimTensor = Input;
        LowDimTensor = this;
    }
    Output = new Tensor(HighDimTensor->shape, HighDimTensor->Device, HighDimTensor->DeviceNum);
    if(Device == "GPU")
    {
        #ifdef CUDA_USEFUL
        AddInCPP(Output->DataGPU, HighDimTensor->DataGPU, HighDimTensor->ShapeCount, LowDimTensor->DataGPU, LowDimTensor->ShapeCount);
        #endif
    }
    else
    {
        for(int a=0; a<HighDimTensor->ShapeCount;a++)
        {
            int ResIndex = a%LowDimTensor->ShapeCount;
            int BlockIndex = int(a/LowDimTensor->ShapeCount);
            Output->DataCPU[a] = HighDimTensor->DataCPU[a] + LowDimTensor->DataCPU[ResIndex];
        }
    }
    return Output;
}

Tensor* Tensor::EleMul(Tensor* Input)
{
    Tensor *Output, *HighDimTensor, *LowDimTensor;
    if(shape.size() > Input->shape.size())
    {
        HighDimTensor = this;
        LowDimTensor = Input;
    }
    else
    {
        HighDimTensor = Input;
        LowDimTensor = this;
    }
    Output = new Tensor(HighDimTensor->shape, HighDimTensor->Device, HighDimTensor->DeviceNum);
    if(Device == "GPU")
    {
        #ifdef CUDA_USEFUL
        EleMulInCPP(Output->DataGPU, HighDimTensor->DataGPU, HighDimTensor->ShapeCount, LowDimTensor->DataGPU, LowDimTensor->ShapeCount);
        #endif
    }
    else
    {
        for(int a=0; a<HighDimTensor->ShapeCount;a++)
        {
            int ResIndex = a%LowDimTensor->ShapeCount;
            int BlockIndex = int(a/LowDimTensor->ShapeCount);
            Output->DataCPU[a] = HighDimTensor->DataCPU[a] * LowDimTensor->DataCPU[ResIndex];
        }
    }
    return Output;
}

size_t Tensor::GetIndex(std::vector<size_t> FindIndex)
{
    size_t _GetIndex = 0;
    size_t ShapeCountTMP = 1;
    for(int a = shape.size() - 1;a>=0;a--)
    {
        _GetIndex += (FindIndex[a])*ShapeCountTMP;
        ShapeCountTMP *= shape[a];
    }
    return _GetIndex;
}

float Tensor::GetV(std::vector<size_t> FindIndex)
{
    float ReturnValue;
    bool GPUflag = 0;
    if(Device == "GPU")
    {
        #ifdef CUDA_USEFUL
        GPUflag = 1;
        ToCPU();
        #endif
    }
    ReturnValue = DataCPU[GetIndex(FindIndex)];
    if(GPUflag)ToGPU();
    return ReturnValue;
}

void Tensor::SetV(std::vector<size_t> FindIndex, float Value)
{
    bool GPUflag = 0;
    if(Device == "GPU")
    {
        #ifdef CUDA_USEFUL
        GPUflag = 1;
        ToCPU();
        #endif
    }
    DataCPU[GetIndex(FindIndex)] = Value;
    #ifdef CUDA_USEFUL
    if(GPUflag)ToGPU();
    #endif
}

CudaDimVec Tensor::TransformFromStdVector(std::vector<size_t> InputVector, size_t ShapeLen)
{
    CudaDimVec ReturenV;
    size_t MinusRes = ShapeLen - InputVector.size();
    memset(ReturenV.Shape,0,sizeof(CudaDimVec));
    for(int a=0;a<InputVector.size();a++)ReturenV.Shape[a+MinusRes] = InputVector[a];
    return ReturenV;
}

Tensor* Tensor::Matmul(Tensor* Input)
{
    Tensor* Output;
    if(shape.size() == 1)
    {
        if(Input->shape.size() == 1)
        {
            Output = new Tensor(std::vector<size_t>{});
            Output->DataCPU[0] = 0;
            if(Device == "GPU")
            {
                #ifdef CUDA_USEFUL
                Output->ToGPU();
                DotArrayInCPP(Output->DataGPU, DataGPU, Input->DataGPU, ShapeCount);
                #endif
            }
            else for(int a=0;a<ShapeCount;a++)Output->DataCPU[0] += DataCPU[a]*Input->DataCPU[a];
        }
        else
        {
            shape = std::vector<size_t>{1,shape[0]};
            Output = Matmul(Input);
            shape = std::vector<size_t>{shape[1]};
            Output->shape = std::vector<size_t>{Output->shape[1]};
        }
    }
    else
    {
        if(Input->shape.size() == 1) 
        {
            Input->shape = std::vector<size_t>{Input->shape[0],1};
            Output = Matmul(Input);
            Input->shape = std::vector<size_t>{Input->shape[0]};
            Output->shape = std::vector<size_t>{Output->shape[0]};
        }
        else
        {
            std::vector<size_t> OutputShape;
            for(int a=0;a<std::max(shape.size(), Input->shape.size())-2;a++)
            {
                if(shape.size() > Input->shape.size())
                {
                    size_t MinusSize = shape.size() - Input->shape.size();
                    if(a < MinusSize)OutputShape.push_back(shape[a]);
                    else OutputShape.push_back(std::max(shape[a], Input->shape[a - MinusSize]));
                }
                else
                {
                    size_t MinusSize = -shape.size() + Input->shape.size();
                    if(a < MinusSize)OutputShape.push_back(Input->shape[a]);
                    else OutputShape.push_back(std::max(shape[a- MinusSize], Input->shape[a]));
                }
            }
            size_t OutputMatrixShape[2] = {shape[shape.size()-2], Input->shape[Input->shape.size()-1]};
            OutputShape.push_back(shape[shape.size()-2]);
            OutputShape.push_back(Input->shape[Input->shape.size()-1]);
            Output = new Tensor(OutputShape, Device, DeviceNum);//OutputShape is all shape of the mat but we only use first 6 elems as the cuda shape params
            Output->FillArray(0.);
            CudaDimVec OutputShapeArray = TransformFromStdVector(OutputShape, Output->shape.size());
            CudaDimVec InputFirstArray = TransformFromStdVector(shape, Output->shape.size());
            CudaDimVec InputSecondArray = TransformFromStdVector(Input->shape, Output->shape.size());
            size_t InputFirstMatrixShape[2] = {shape[shape.size()-2], shape[shape.size()-1]};
            size_t InputSecondMatrixShape[2] = {Input->shape[Input->shape.size()-2], Input->shape[Input->shape.size()-1]};
            if(Device == "GPU")
            {
                #ifdef CUDA_USEFUL
                MatmulInCPP
                (
                    Output->DataGPU,
                    OutputShapeArray.Shape, 
                    OutputMatrixShape,
                    DataGPU, 
                    InputFirstArray.Shape,
                    InputFirstMatrixShape,
                    Input->DataGPU,
                    InputSecondArray.Shape,
                    InputSecondMatrixShape,
                    Output->shape.size()-2,
                    Output->ShapeCount,
                    DeviceNum
                );
                #endif
            }
            else
            {
                size_t BatchShapeLen = Output->shape.size()-2;
                size_t *OutputBatchShape = OutputShapeArray.Shape;
                size_t *InputFirstBatchShape = InputFirstArray.Shape;
                size_t *InputSecondBatchShape = InputSecondArray.Shape;
                float *InputFirst = DataCPU;
                float *InputSecond = Input->DataCPU;
                for(int Index = 0;Index < Output->ShapeCount;Index++)
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
                    Output->DataCPU[Index] = 0;
                    for(int a=0;a<InputFirstMatrixShape[1];a++)
                    {
                      Output->DataCPU[Index] += InputFirst[InputFirstBase + MatIndex[0]*InputFirstMatrixShape[1] + a]*InputSecond[InputSecondBase + a*InputSecondMatrixShape[1] + MatIndex[1]];
                    }
                }
            }
        }
    }
    return Output;
}

Tensor* Tensor::T()
{
    Tensor* Output = new Tensor(shape,Device, DeviceNum);
    size_t TMP = Output->shape[Output->shape.size() - 1];
    Output->shape[Output->shape.size() - 1] = Output->shape[Output->shape.size() - 2];
    Output->shape[Output->shape.size() - 2] = TMP;
    size_t InputFirstMatrixShape[2] = {shape[shape.size()-2], shape[shape.size()-1]};
    if(Device == "GPU")
    {
        #ifdef CUDA_USEFUL
        TInCPP(Output->DataGPU, DataGPU, InputFirstMatrixShape,ShapeCount);
        #endif
    }
    else
    {
        size_t *MatrixShape = InputFirstMatrixShape;
        for(int Index = 0 ;Index < ShapeCount;Index++)
        {
            size_t MatrixShapeCount = MatrixShape[0]*MatrixShape[1];
            size_t InputMatIndex = Index%MatrixShapeCount;
            size_t BaseCount = Index - InputMatIndex;
            size_t InputMatIndexFirst = InputMatIndex/MatrixShape[1];
            size_t InputMatIndexSecond = InputMatIndex%MatrixShape[1];
            Output->DataCPU[BaseCount + InputMatIndexSecond*MatrixShape[0] + InputMatIndexFirst] = DataCPU[Index];
        }
    }
    return Output;
}

Tensor* Tensor::SumTensorDim(size_t InputDim)
{
    std::vector<size_t>OutputShape;
    for(int a=0;a<shape.size();a++)
    {
        if(a!=InputDim)OutputShape.push_back(shape[a]);
        else OutputShape.push_back(1);
    }
    Tensor *Output = new Tensor(OutputShape, Device, DeviceNum);
    CudaDimVec ShapeArray = TransformFromStdVector(shape, shape.size());
    if(Device == "GPU")
    {
        #ifdef CUDA_USEFUL
        SumTensorDimInCPP(Output->DataGPU, DataGPU, ShapeArray.Shape,OutputShape.size(),InputDim, Output->ShapeCount);
        #endif
    }
    else
    {
        size_t InputShapeLen = OutputShape.size();
        size_t *InputShape = ShapeArray.Shape;
        for(int Index = 0 ;Index < Output->ShapeCount;Index++)
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
            Output->DataCPU[Index] = 0;
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
              Output->DataCPU[Index] += DataCPU[InputDimIndex];
            }
        }
    }

    return Output;
}

Tensor* Tensor::AverageTensorDim(size_t InputDim)
{
    return SumTensorDim(InputDim)->MulScalar(1./(shape[InputDim]));
}

void Tensor::GaussianElimination()
{
    size_t BatchSize = 1;
    for(int a=0;a<shape.size() - 2;a++)
    {
        BatchSize*=shape[a];
    }
    //求逆只需要最后两维度作为矩阵,其他的都是batch

    if(Device == "GPU")
    {
        #ifdef CUDA_USEFUL
        //todo
        ToCPU();
        GaussianElimination();
        ToGPU();
        #endif
    }
    else
    {
        std::vector<std::thread>ThreadList;
        for(int a=0;a<BatchSize;a++)
        {
            ThreadList.push_back(std::move(std::thread(MatrixGaussianElimination,DataCPU+a*shape[shape.size()-2]*shape[shape.size()-1],shape[shape.size()-2], shape[shape.size()-1])));
        }
        for(int a=0;a<ThreadList.size();a++)
        {
            ThreadList[a].join();
        }
    }
}

Tensor* Tensor::TensorSplice(Tensor* InputTensor, int SpliceDim)
{
    std::vector<size_t>ReturnShape;
    for(int a=0; a<shape.size();a++)
    {
        ReturnShape.push_back(shape[a] + InputTensor->shape[a]*(a == SpliceDim));
    }
    Tensor* ReturnTensor = new Tensor(ReturnShape, Device, DeviceNum);
    CudaDimVec ShapeArraySelf = TransformFromStdVector(shape, shape.size());
    CudaDimVec ShapeArrayFirst = TransformFromStdVector(InputTensor->shape, InputTensor->shape.size());
    if(Device == "GPU")
    {
        #ifdef CUDA_USEFUL
        TensorSpliceInCPP(ReturnTensor->DataGPU , DataGPU, InputTensor->DataGPU, ShapeArraySelf.Shape, ShapeArrayFirst.Shape, shape.size(), SpliceDim, ReturnTensor->ShapeCount);
        #endif
    }
    else
    {
        size_t InputDim = SpliceDim;
        size_t InputShapeLen = shape.size();
        size_t* InputShapeFirst = ShapeArraySelf.Shape;
        size_t* InputShapeSecond = ShapeArrayFirst.Shape;
        float* OutputData = ReturnTensor->DataCPU;
        float* InputDataFirst = DataCPU;
        float* InputDataSecond = InputTensor->DataCPU;
        for(int Index=0;Index<ReturnTensor->ShapeCount;Index++)
        {
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
    }
    return ReturnTensor;
}
