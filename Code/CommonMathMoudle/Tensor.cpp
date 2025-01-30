#include "Tensor.h"

Tensor::Tensor(std::vector<size_t>shape)
{
    InitTensor(shape,0);
}

Tensor::Tensor(std::vector<size_t>shape, size_t DeviceNum)
{
    InitTensor(shape,DeviceNum);
}

Tensor::Tensor(std::vector<size_t>shape, size_t DeviceNum, std::vector<float> InputData)
{
    InitTensor(shape,0);
    for(size_t a = 0;a<InputData.size();a++)
    {
        GetDevicePointer()[a] = InputData[a];
    }
    if(DeviceNum)ToDevice(DeviceNum);
}

Tensor::Tensor(std::vector<size_t>shape, size_t DeviceNum, std::vector<float>* InputData)
{
    InitTensor(shape,0);
    for(size_t a = 0;a<(*InputData).size();a++)
    {
        GetDevicePointer()[a] = (*InputData)[a];
    }
    if(DeviceNum)ToDevice(DeviceNum);
}

Tensor* Tensor::CreateTensorByLoadPath(std::ifstream& OpenedFile, size_t DeviceNum)
{
    Tensor* ReturnTensor = Tensor::CreateTensorByLoadPath(OpenedFile);
    ReturnTensor->ToDevice(DeviceNum);
    return ReturnTensor;
}

Tensor* Tensor::CreateTensorByLoadPath(std::ifstream& OpenedFile)
{
    Tensor* ReturnTensor = new Tensor();
    ReturnTensor->LoadFromFile(OpenedFile);
    return ReturnTensor;
}

Tensor* Tensor::CreateTensorByLoadPath(std::string LoadPath, size_t DeviceNum)
{
    Tensor* ReturnTensor = Tensor::CreateTensorByLoadPath(LoadPath);
    ReturnTensor->ToDevice(DeviceNum);
    return ReturnTensor;
}

Tensor* Tensor::CreateTensorByLoadPath(std::string LoadPath)
{
    Tensor* ReturnTensor = new Tensor();
    ReturnTensor->LoadFromFile(LoadPath);
    return ReturnTensor;
}

Tensor* Tensor::CopyNewEmptyTensor()
{
    return new Tensor(shape, GetDeviceNum());
}

Tensor* Tensor::Copy()
{
    return this->AddScalar(0);
}

void Tensor::InitTensor(std::vector<size_t>shape, size_t DeviceNum)
{
    this->shape = shape;
    ShapeCount = 1;
    for(int a=0;a<shape.size();a++)ShapeCount*=shape[a];
    DPMgr = std::make_shared<DevicePointerManager>(DeviceNum, ShapeCount);
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
    size_t ProtoDeviceNum = GetDeviceNum();
    ToDevice(0);
    float* DataPointer = GetDevicePointer();
    for(int a=0;a<ShapeCount;a++)
    {
        std::cout<< std::fixed << std::setprecision(12) <<DataPointer[a]<<" ";
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
    if(shape.size()<2)std::cout<<std::endl;
    ToDevice(ProtoDeviceNum);
}

void Tensor::PrintShape()
{
    std::cout<<"Shape:{";
    for(size_t a=0;a<shape.size();a++)
    {
        std::cout<<" "<<shape[a];
    }
    std::cout<<" }"<<std::endl;
}

void Tensor::FillArray(float Scalar)
{
    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        FillArrayInCPP(GetDevicePointer(), Scalar, ShapeCount);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().FillArrayInCPP, 
            ShapeCount, 
            {
                GetDeviceBuffer(),
                VBuffer::CVBuffer(Scalar).OpenGLTMPBuffer, 
                VBuffer::CVBuffer((int)ShapeCount).OpenGLTMPBuffer
            }
        );
        #endif
    }
    else
    {
        for(int a=0;a<ShapeCount;a++)
        {
            GetDevicePointer()[a] = Scalar;
        }
    }
}

Tensor* Tensor::AddScalar(float Scalar)
{
    Tensor* Output = new Tensor(shape, GetDeviceNum());
    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        AddScalarInCPP(Output->GetDevicePointer(), GetDevicePointer(), Scalar, ShapeCount);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().AddScalarInCPP, 
            ShapeCount, 
            {
                Output->GetDeviceBuffer(),
                GetDeviceBuffer(),
                VBuffer::CVBuffer(Scalar).OpenGLTMPBuffer, 
                VBuffer::CVBuffer((int)ShapeCount).OpenGLTMPBuffer
            }
        );
        #endif
    }
    else for(int a=0;a<ShapeCount;a++)Output->GetDevicePointer()[a] = GetDevicePointer()[a]+ Scalar;
    return Output;
}

Tensor* Tensor::MulScalar(float Scalar)
{
    Tensor* Output = new Tensor(shape, GetDeviceNum());
    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        MulScalarInCPP(Output->GetDevicePointer(), GetDevicePointer(), Scalar, ShapeCount);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().MulScalarInCPP, 
            ShapeCount, 
            {
                Output->GetDeviceBuffer(),
                GetDeviceBuffer(),
                VBuffer::CVBuffer(Scalar).OpenGLTMPBuffer, 
                VBuffer::CVBuffer((int)ShapeCount).OpenGLTMPBuffer
            }
        );
        #endif
    }
    else for(int a=0;a<ShapeCount;a++)Output->GetDevicePointer()[a] = GetDevicePointer()[a] *Scalar;
    return Output;
}

Tensor* Tensor::AddArray(Tensor* Input)
{
    Tensor* Output = new Tensor(shape, GetDeviceNum());
    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        AddArrayInCPP(Output->GetDevicePointer(), GetDevicePointer(), Input->GetDevicePointer(), ShapeCount);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().AddArrayInCPP, 
            ShapeCount, 
            {
                Output->GetDeviceBuffer(),
                GetDeviceBuffer(),
                Input->GetDeviceBuffer(), 
                VBuffer::CVBuffer((int)ShapeCount).OpenGLTMPBuffer
            }
        );
        #endif
    }
    else for(int a=0;a<ShapeCount;a++)Output->GetDevicePointer()[a] = GetDevicePointer()[a] + Input->GetDevicePointer()[a];
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
    Output = new Tensor(HighDimTensor->shape, HighDimTensor->GetDeviceNum());
    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        AddInCPP(Output->GetDevicePointer(), HighDimTensor->GetDevicePointer(), HighDimTensor->ShapeCount, LowDimTensor->GetDevicePointer(), LowDimTensor->ShapeCount);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().AddInCPP, 
            Output->ShapeCount,
            {
                Output->GetDeviceBuffer(),
                HighDimTensor->GetDeviceBuffer(),
                VBuffer::CVBuffer((int)(HighDimTensor->ShapeCount)).OpenGLTMPBuffer,
                LowDimTensor->GetDeviceBuffer(),
                VBuffer::CVBuffer((int)(LowDimTensor->ShapeCount)).OpenGLTMPBuffer
            }
        );
        #endif 
    }
    else
    {
        for(int a=0; a<HighDimTensor->ShapeCount;a++)
        {
            int ResIndex = a%LowDimTensor->ShapeCount;
            int BlockIndex = int(a/LowDimTensor->ShapeCount);
            Output->GetDevicePointer()[a] = HighDimTensor->GetDevicePointer()[a] + LowDimTensor->GetDevicePointer()[ResIndex];
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
    Output = new Tensor(HighDimTensor->shape, HighDimTensor->GetDeviceNum());
    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        EleMulInCPP(Output->GetDevicePointer(), HighDimTensor->GetDevicePointer(), HighDimTensor->ShapeCount, LowDimTensor->GetDevicePointer(), LowDimTensor->ShapeCount);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().EleMulInCPP, 
            Output->ShapeCount,
            {
                Output->GetDeviceBuffer(),
                HighDimTensor->GetDeviceBuffer(),
                VBuffer::CVBuffer((int)(HighDimTensor->ShapeCount)).OpenGLTMPBuffer,
                LowDimTensor->GetDeviceBuffer(),
                VBuffer::CVBuffer((int)(LowDimTensor->ShapeCount)).OpenGLTMPBuffer
            }
        );
        #endif
    }
    else
    {
        for(int a=0; a<HighDimTensor->ShapeCount;a++)
        {
            int ResIndex = a%LowDimTensor->ShapeCount;
            int BlockIndex = int(a/LowDimTensor->ShapeCount);
            Output->GetDevicePointer()[a] = HighDimTensor->GetDevicePointer()[a] * LowDimTensor->GetDevicePointer()[ResIndex];
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
    size_t ProtoDeviceNum = GetDeviceNum();
    ToDevice(0);
    ReturnValue = GetDevicePointer()[GetIndex(FindIndex)];
    ToDevice(ProtoDeviceNum);
    return ReturnValue;
}

void Tensor::SetV(std::vector<size_t> FindIndex, float Value)
{
    size_t ProtoDeviceNum = GetDeviceNum();
    ToDevice(0);
    GetDevicePointer()[GetIndex(FindIndex)] = Value;
    ToDevice(ProtoDeviceNum);
}

CudaDimVec Tensor::TransformFromStdVector(std::vector<size_t> InputVector, size_t ShapeLen)
{
    CudaDimVec ReturenV;
    size_t MinusRes = ShapeLen - InputVector.size();
    for(int a=0;a<8;a++)ReturenV.Shape[a]=0;
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
            Output->GetDevicePointer()[0] = 0;
            if(GetDeviceNum())
            {
                #ifdef CUDA_USEFUL
                Output->ToDevice(GetDeviceNum());
                DotArrayInCPP(Output->GetDevicePointer(), GetDevicePointer(), Input->GetDevicePointer(), ShapeCount);
                #endif
                #ifdef OPENGL_USEFUL
                Log::Assert(false, "OpenGL::Matmul_1::todo");
                #endif
            }
            else for(int a=0;a<ShapeCount;a++)Output->GetDevicePointer()[0] += GetDevicePointer()[a]*Input->GetDevicePointer()[a];
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
            Output = new Tensor(OutputShape, GetDeviceNum());//OutputShape is all shape of the mat but we only use first 6 elems as the cuda shape params
            Output->FillArray(0.);
            CudaDimVec OutputShapeArray = TransformFromStdVector(OutputShape, Output->shape.size());
            CudaDimVec InputFirstArray = TransformFromStdVector(shape, Output->shape.size());
            CudaDimVec InputSecondArray = TransformFromStdVector(Input->shape, Output->shape.size());
            size_t InputFirstMatrixShape[2] = {shape[shape.size()-2], shape[shape.size()-1]};
            size_t InputSecondMatrixShape[2] = {Input->shape[Input->shape.size()-2], Input->shape[Input->shape.size()-1]};
            if(GetDeviceNum())
            {
                #ifdef CUDA_USEFUL
                MatmulInCPP
                (
                    Output->GetDevicePointer(),
                    OutputShapeArray.Shape, 
                    OutputMatrixShape,
                    GetDevicePointer(), 
                    InputFirstArray.Shape,
                    InputFirstMatrixShape,
                    Input->GetDevicePointer(),
                    InputSecondArray.Shape,
                    InputSecondMatrixShape,
                    Output->shape.size()-2,
                    Output->ShapeCount,
                    DeviceNumToCuda(GetDeviceNum())
                );
                #endif
                #ifdef OPENGL_USEFUL
                int OutputMatrixShape_int[2] = {(int)(shape[shape.size()-2]), (int)(Input->shape[Input->shape.size()-1])};
                int InputFirstMatrixShape_int[2] = {(int)(shape[shape.size()-2]), (int)(shape[shape.size()-1])};
                int InputSecondMatrixShape_int[2] = {(int)(Input->shape[Input->shape.size()-2]), (int)(Input->shape[Input->shape.size()-1])};
                GPUDeviceProcess::I().ProcessGLSLFun
                (
                    GLSL::I().MatmulInCPP, 
                    Output->ShapeCount,
                    {
                        Output->GetDeviceBuffer(),
                        VBuffer::CVBuffer(OutputShapeArray.ToInt(), OutputShapeArray.ShapeLen).OpenGLTMPBuffer,
                        VBuffer::CVBuffer(OutputMatrixShape_int, 2).OpenGLTMPBuffer,
                        GetDeviceBuffer(),
                        VBuffer::CVBuffer(InputFirstArray.ToInt(), InputFirstArray.ShapeLen).OpenGLTMPBuffer,
                        VBuffer::CVBuffer(InputFirstMatrixShape_int, 2).OpenGLTMPBuffer,
                        Input->GetDeviceBuffer(),
                        VBuffer::CVBuffer(InputSecondArray.ToInt(), InputSecondArray.ShapeLen).OpenGLTMPBuffer,
                        VBuffer::CVBuffer(InputSecondMatrixShape_int, 2).OpenGLTMPBuffer,
                        VBuffer::CVBuffer((int)(Output->shape.size()-2)).OpenGLTMPBuffer,
                        VBuffer::CVBuffer((int)(Output->ShapeCount)).OpenGLTMPBuffer,
                    }
                );
                #endif
            }
            else
            {
                size_t BatchShapeLen = Output->shape.size()-2;
                size_t *OutputBatchShape = OutputShapeArray.Shape;
                size_t *InputFirstBatchShape = InputFirstArray.Shape;
                size_t *InputSecondBatchShape = InputSecondArray.Shape;
                float *InputFirst = GetDevicePointer();
                float *InputSecond = Input->GetDevicePointer();
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
                    Output->GetDevicePointer()[Index] = 0;
                    for(int a=0;a<InputFirstMatrixShape[1];a++)
                    {
                      Output->GetDevicePointer()[Index] += InputFirst[InputFirstBase + MatIndex[0]*InputFirstMatrixShape[1] + a]*InputSecond[InputSecondBase + a*InputSecondMatrixShape[1] + MatIndex[1]];
                    }
                }
            }
        }
    }
    return Output;
}

Tensor* Tensor::T()
{
    Tensor* Output = new Tensor(shape,GetDeviceNum());
    size_t TMP = Output->shape[Output->shape.size() - 1];
    Output->shape[Output->shape.size() - 1] = Output->shape[Output->shape.size() - 2];
    Output->shape[Output->shape.size() - 2] = TMP;
    size_t InputFirstMatrixShape[2] = {shape[shape.size()-2], shape[shape.size()-1]};
    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        TInCPP(Output->GetDevicePointer(), GetDevicePointer(), InputFirstMatrixShape,ShapeCount);
        #endif
        #ifdef OPENGL_USEFUL
        int InputFirstMatrixShape_int[2] = {(int)(shape[shape.size()-2]), (int)(shape[shape.size()-1])};
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().TInCPP, 
            Output->ShapeCount,
            {
                Output->GetDeviceBuffer(),
                GetDeviceBuffer(),
                VBuffer::CVBuffer(InputFirstMatrixShape_int, 2).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(Output->ShapeCount)).OpenGLTMPBuffer,
            }
        );
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
            Output->GetDevicePointer()[BaseCount + InputMatIndexSecond*MatrixShape[0] + InputMatIndexFirst] = GetDevicePointer()[Index];
        }
    }
    return Output;
}

Tensor* Tensor::MaximumOrMinimum(size_t InputDim, bool IsMaximum)
{
    std::vector<size_t>OutputShape;
    for(int a=0;a<shape.size();a++)
    {
        if(a!=InputDim)OutputShape.push_back(shape[a]);
        else OutputShape.push_back(1);
    }
    Tensor *Output = new Tensor(OutputShape, GetDeviceNum());
    CudaDimVec ShapeArray = TransformFromStdVector(shape, shape.size());
    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        MaximumOrMinimumTensorDimInCPP(Output->GetDevicePointer(), GetDevicePointer(), ShapeArray.Shape,OutputShape.size(),InputDim, Output->ShapeCount, IsMaximum);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().MaximumOrMinimumTensorDimInCPP, 
            Output->ShapeCount,
            {
                Output->GetDeviceBuffer(),
                GetDeviceBuffer(),
                VBuffer::CVBuffer(ShapeArray.ToInt(), ShapeArray.ShapeLen).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(OutputShape.size())).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(InputDim)).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(Output->ShapeCount)).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(IsMaximum)).OpenGLTMPBuffer,
            }
        );
        #endif
    }
    else
    {
        size_t InputShapeLen = OutputShape.size();
        size_t *InputShape = ShapeArray.Shape;
        for(int Index = 0 ;Index < Output->ShapeCount;Index++)
        {
          if(IsMaximum)
          {
            Output->GetDevicePointer()[Index] = -1e9+7;
          }
          else
          {
            Output->GetDevicePointer()[Index] = 1e9+7;
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
              Output->GetDevicePointer()[Index] = std::max(Output->GetDevicePointer()[Index], GetDevicePointer()[InputDimIndex]);
            }
            else
            {
              Output->GetDevicePointer()[Index] = std::min(Output->GetDevicePointer()[Index], GetDevicePointer()[InputDimIndex]);
            }
          }
        }
    }
    return Output;
}

Tensor* Tensor::Minimum(size_t InputDim)
{
    return MaximumOrMinimum(InputDim, false);
}

Tensor* Tensor::Maximum(size_t InputDim)
{
    return MaximumOrMinimum(InputDim, true);
}

Tensor* Tensor::Mean(std::vector<size_t>InputDims)
{
    Tensor* TMPTensor = Sum(InputDims);
    float SumDimRes = 1;
    for(size_t a = 0;a<InputDims.size();a++)
    {
        SumDimRes*=shape[InputDims[a]];
    }
    Tensor* ResTensor = TMPTensor->MulScalar(1./SumDimRes);
    delete TMPTensor;
    return ResTensor;
}

//Tensor* Tensor::Var(std::vector<size_t>InputDims)
//{
//    Tensor* MeanTensor = Mean(InputDims);
//    //todo
//}

Tensor* Tensor::Sum(std::vector<size_t>InputDims)
{
    Tensor* ResTensor = SumTensorDim(InputDims[0]);
    for(size_t a = 1;a<InputDims.size();a++)
    {
        Tensor* TMPTensor = ResTensor->SumTensorDim(InputDims[a]);
        delete ResTensor;
        ResTensor = TMPTensor;
    }
    return ResTensor;
}

Tensor* Tensor::SumTensorDim(size_t InputDim)
{
    std::vector<size_t>OutputShape;
    for(int a=0;a<shape.size();a++)
    {
        if(a!=InputDim)OutputShape.push_back(shape[a]);
        else OutputShape.push_back(1);
    }
    Tensor *Output = new Tensor(OutputShape, GetDeviceNum());
    CudaDimVec ShapeArray = TransformFromStdVector(shape, shape.size());
    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        SumTensorDimInCPP(Output->GetDevicePointer(), GetDevicePointer(), ShapeArray.Shape,OutputShape.size(),InputDim, Output->ShapeCount);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().SumTensorDimInCPP, 
            Output->ShapeCount,
            {
                Output->GetDeviceBuffer(),
                GetDeviceBuffer(),
                VBuffer::CVBuffer(ShapeArray.ToInt(), ShapeArray.ShapeLen).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(OutputShape.size())).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(InputDim)).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(Output->ShapeCount)).OpenGLTMPBuffer
            }
        );
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
            Output->GetDevicePointer()[Index] = 0;
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
              Output->GetDevicePointer()[Index] += GetDevicePointer()[InputDimIndex];
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

    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        GaussianEliminationInCPP(GetDevicePointer(), BatchSize, shape[shape.size()-2], shape[shape.size()-1]);
        #endif
        #ifdef OPENGL_USEFUL
        Log::Assert(false, "OpenGL::GaussianElimination::todo");
        #endif
    }
    else
    {
        #ifdef THREAD_USEFUL
        std::vector<std::thread>ThreadList;
        #endif
        for(int a=0;a<BatchSize;a++)
        {
            #ifdef THREAD_USEFUL
            ThreadList.push_back(std::move(std::thread(MatrixGaussianElimination,GetDevicePointer()+a*shape[shape.size()-2]*shape[shape.size()-1],shape[shape.size()-2], shape[shape.size()-1])));
            #else
            MatrixGaussianElimination(GetDevicePointer()+a*shape[shape.size()-2]*shape[shape.size()-1],shape[shape.size()-2], shape[shape.size()-1]);
            #endif
        }
        #ifdef THREAD_USEFUL
        for(int a=0;a<ThreadList.size();a++)
        {
            ThreadList[a].join();
        }
        #endif
    }
}

Tensor* Tensor::TensorSplice(Tensor* InputTensor, int SpliceDim)
{
    std::vector<size_t>ReturnShape;
    for(int a=0; a<shape.size();a++)
    {
        ReturnShape.push_back(shape[a] + InputTensor->shape[a]*(a == SpliceDim));
    }
    Tensor* ReturnTensor = new Tensor(ReturnShape, GetDeviceNum());
    CudaDimVec ShapeArraySelf = TransformFromStdVector(shape, shape.size());
    CudaDimVec ShapeArrayFirst = TransformFromStdVector(InputTensor->shape, InputTensor->shape.size());
    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        TensorSpliceInCPP(ReturnTensor->GetDevicePointer() , GetDevicePointer(), InputTensor->GetDevicePointer(), ShapeArraySelf.Shape, ShapeArrayFirst.Shape, shape.size(), SpliceDim, ReturnTensor->ShapeCount);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().TensorSpliceInCPP, 
            ReturnTensor->ShapeCount,
            {
                ReturnTensor->GetDeviceBuffer(),
                GetDeviceBuffer(),
                InputTensor->GetDeviceBuffer(),
                VBuffer::CVBuffer(ShapeArraySelf.ToInt(), ShapeArraySelf.ShapeLen).OpenGLTMPBuffer,
                VBuffer::CVBuffer(ShapeArrayFirst.ToInt(), ShapeArrayFirst.ShapeLen).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(shape.size())).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(SpliceDim)).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(ReturnTensor->ShapeCount)).OpenGLTMPBuffer,
            }
        );
        #endif
    }
    else
    {
        size_t InputDim = SpliceDim;
        size_t InputShapeLen = shape.size();
        size_t* InputShapeFirst = ShapeArraySelf.Shape;
        size_t* InputShapeSecond = ShapeArrayFirst.Shape;
        float* OutputData = ReturnTensor->GetDevicePointer();
        float* InputDataFirst = GetDevicePointer();
        float* InputDataSecond = InputTensor->GetDevicePointer();
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

Tensor* Tensor::GetUnitTensor(std::vector<size_t>ReturnShape, size_t ReturnDeviceNum)
{
    Tensor* ReturnTensor = new Tensor(ReturnShape, ReturnDeviceNum);
    CudaDimVec ShapeArray = ReturnTensor->TransformFromStdVector(ReturnShape, ReturnShape.size());
    ReturnTensor->FillArray(0.);
    if(ReturnDeviceNum)
    {
        #ifdef CUDA_USEFUL
        GetUnitTensorInCPP(ReturnTensor->GetDevicePointer(), ShapeArray.Shape, ReturnTensor->ShapeCount, ReturnTensor->shape.size());
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().GetUnitTensorInCPP, 
            ReturnTensor->ShapeCount,
            {
                ReturnTensor->GetDeviceBuffer(),
                VBuffer::CVBuffer(ShapeArray.ToInt(), ShapeArray.ShapeLen).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(ReturnTensor->ShapeCount)).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(ReturnTensor->shape.size())).OpenGLTMPBuffer, 
            }
        );
        #endif
    }
    else
    {
        size_t* InputShape = ShapeArray.Shape;
        size_t InputShapeLen =  ReturnTensor->shape.size();
        size_t OutputShapeCount = ReturnTensor->ShapeCount;
        float*OutputData =  ReturnTensor->GetDevicePointer();
        for(int a=0;a<ReturnTensor->ShapeCount;a++)
        {
            size_t Index = a;
            size_t MatrixShapeCount = InputShape[InputShapeLen - 2]*InputShape[InputShapeLen - 1];
            size_t MatrixIndex = Index%MatrixShapeCount;
            if(MatrixIndex%InputShape[InputShapeLen - 2] == MatrixIndex/InputShape[InputShapeLen - 2])
            {
              OutputData[Index] = 1;
            }
        }
    }
    return ReturnTensor;
}

Tensor* Tensor::GetTensorBy2ShapeVector(std::vector<size_t>StartShapeVector, std::vector<size_t>EndShapeVector)
{
    std::vector<size_t>ReturnTensorShape;
    for(int a=0;a<StartShapeVector.size();a++)
    {
        ReturnTensorShape.push_back(EndShapeVector[a] - StartShapeVector[a] + 1);
    }
    Tensor* ReturnTensor = new Tensor(ReturnTensorShape,GetDeviceNum());
    CudaDimVec StartShapeArray = TransformFromStdVector(StartShapeVector, StartShapeVector.size());
    CudaDimVec EndShapeArray = TransformFromStdVector(EndShapeVector, EndShapeVector.size());
    CudaDimVec InputShapeArray = TransformFromStdVector(shape, shape.size());
    CudaDimVec OutputShapeArray = TransformFromStdVector(ReturnTensorShape, ReturnTensorShape.size());
    size_t* InputShape = InputShapeArray.Shape;
    size_t* OutputShape = OutputShapeArray.Shape;
    size_t* StartShape = StartShapeArray.Shape;
    size_t* EndShape = EndShapeArray.Shape;
    size_t ShapeLen = shape.size();

    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        GetTensorBy2ShapeVectorInCPP(ReturnTensor->GetDevicePointer(), GetDevicePointer(), InputShape,OutputShape,StartShape, EndShape, ShapeLen);
        #endif
        #ifdef OPENGL_USEFUL
        Log::Assert(false, "OpenGL::GetTensorBy2ShapeVector::todo");
        #endif
    }
    else
    {
        float* OutputData = ReturnTensor->GetDevicePointer();
        float* InputData = GetDevicePointer();
        for(int Index = 0;Index < ReturnTensor->ShapeCount;Index++)
        {
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
    }
    return ReturnTensor;
}

Tensor* Tensor::Inverse()
{
    Tensor* ReturnUnit = GetUnitTensor(shape, GetDeviceNum());
    Tensor* SpliceTensor = TensorSplice(ReturnUnit, shape.size() - 1);
    SpliceTensor->GaussianElimination();
    std::vector<size_t>StartShape, EndShape;
    for(int a=0;a<shape.size();a++)
    {
        EndShape.push_back(SpliceTensor->shape[a] - 1);
        if(a<shape.size()-1)
        {
            StartShape.push_back(0);
        }
        else
        {
            StartShape.push_back(SpliceTensor->shape[a]/2);
        }
    }
    Tensor* ReturnTensor = SpliceTensor->GetTensorBy2ShapeVector(StartShape, EndShape);
    delete ReturnUnit;
    delete SpliceTensor;
    return ReturnTensor;
}

Tensor* Tensor::EleExp(float BaseNum)
{
    Tensor* ReturnTensor = Copy();
    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        EleExpInCPP(ReturnTensor->GetDevicePointer(), ReturnTensor->ShapeCount, BaseNum);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().EleExpInCPP, 
            ReturnTensor->ShapeCount,
            {
                ReturnTensor->GetDeviceBuffer(),
                VBuffer::CVBuffer((int)(ReturnTensor->ShapeCount)).OpenGLTMPBuffer,
                VBuffer::CVBuffer(BaseNum).OpenGLTMPBuffer, 
            }
        );
        #endif
    }
    else
    {
        for(size_t Index = 0;Index < ReturnTensor->ShapeCount;Index++)
        {
            ReturnTensor->GetDevicePointer()[Index] = powf(BaseNum, ReturnTensor->GetDevicePointer()[Index]);
        }
    }
    return ReturnTensor;
}

bool Tensor::CanBroadCastTo(std::vector<size_t>BroadCastShape)
{
    if(BroadCastShape.size() < shape.size())return false;
    for(int a = 0;a<shape.size();a++)
    {
        if(BroadCastShape[BroadCastShape.size()-a-1]!=shape[shape.size()-a-1] && shape[shape.size()-a-1]!=1)return false;
    }
    return true;
}

Tensor* Tensor::BroadCastTo(std::vector<size_t>BroadCastShape)
{
    Log::Assert(CanBroadCastTo(BroadCastShape), std::string("Broad Cast Shape Not Match"));
    Tensor* ReturnTensor = new Tensor(BroadCastShape, GetDeviceNum());
    std::vector<size_t>FixedShape;
    for(int a=0;;a++)
    {
        if(a+shape.size() < BroadCastShape.size())
        {
            FixedShape.push_back(1);
        }
        else
        {
            if(a >= BroadCastShape.size())
            {
                break;
            }
            FixedShape.push_back(shape[a + shape.size() -BroadCastShape.size()]);
        }
    }
    CudaDimVec FixedShapeArray = TransformFromStdVector(FixedShape, FixedShape.size());
    CudaDimVec OutputShapeArray = TransformFromStdVector(ReturnTensor->shape, ReturnTensor->shape.size());
    size_t* FixedShapePointer = FixedShapeArray.Shape;
    size_t* OutputShapePointer = OutputShapeArray.Shape;
    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        BroadCastToInCPP(ReturnTensor->GetDevicePointer(), GetDevicePointer(), OutputShapePointer, FixedShapePointer, ReturnTensor->shape.size(), ReturnTensor->ShapeCount);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().BroadCastToInCPP, 
            ReturnTensor->ShapeCount,
            {
                ReturnTensor->GetDeviceBuffer(),
                GetDeviceBuffer(),
                VBuffer::CVBuffer(OutputShapeArray.ToInt(), OutputShapeArray.ShapeLen).OpenGLTMPBuffer,
                VBuffer::CVBuffer(FixedShapeArray.ToInt(), FixedShapeArray.ShapeLen).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(ReturnTensor->shape.size())).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(ReturnTensor->ShapeCount)).OpenGLTMPBuffer,
            }
        );
        #endif
    }
    else
    {
        for(size_t Index = 0;Index < ReturnTensor->ShapeCount;Index++)
        {
            size_t ShapeIndex[10];
            size_t NowIndex = Index;
            for(int a = FixedShape.size() - 1 ;a >= 0;a--)
            {
              ShapeIndex[a] = NowIndex%OutputShapePointer[a];
              NowIndex = size_t(NowIndex/OutputShapePointer[a]);
              if(OutputShapePointer[a] > FixedShapePointer[a])ShapeIndex[a] = 0;
            }
            size_t FixedInputIndex = 0;
            for(size_t a = 0;a<FixedShape.size();a++)
            {
              FixedInputIndex *= FixedShapePointer[a];
              FixedInputIndex += ShapeIndex[a];
            }
            ReturnTensor->GetDevicePointer()[Index] = GetDevicePointer()[FixedInputIndex];
        }
    }
    return ReturnTensor;
}

Tensor* Tensor::EleInverse()
{
    Tensor* ReturnTensor = Copy();
    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        EleInverseInCPP(ReturnTensor->GetDevicePointer(), ShapeCount);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().EleInverseInCPP, 
            ShapeCount,
            {
                ReturnTensor->GetDeviceBuffer(),
                VBuffer::CVBuffer((int)(ShapeCount)).OpenGLTMPBuffer,
            }
        );
        #endif
    }
    else
    {
        for(size_t Index = 0;Index < ShapeCount;Index++)
        {
            ReturnTensor->GetDevicePointer()[Index] = 1./ReturnTensor->GetDevicePointer()[Index];
        }
    }
    return ReturnTensor;
}

Tensor* Tensor::Softmax(size_t InputDim)
{
    Tensor* ReturnTensor = Copy();
    Tensor* MaxTensor = ReturnTensor->Maximum(InputDim);
    Tensor* MaxTensorBroadCastTo = MaxTensor->BroadCastTo(ReturnTensor->shape);
    delete MaxTensor;
    Tensor* MaxMinus = MaxTensorBroadCastTo->MulScalar(-1.);
    delete MaxTensorBroadCastTo;
    Tensor* ReturnTensorMinusMaximum = ReturnTensor->Add(MaxMinus);
    delete MaxMinus;
    delete ReturnTensor;
    Tensor* ExpResult = ReturnTensorMinusMaximum->EleExp(M_E);
    Tensor* SumResult = ExpResult->SumTensorDim(InputDim);
    Tensor* SumResultBroadCastTo = SumResult->BroadCastTo(shape);
    delete SumResult;
    Tensor* SumEleInverseResult = SumResultBroadCastTo->EleInverse();
    delete SumResultBroadCastTo;
    ReturnTensor = ExpResult->EleMul(SumEleInverseResult);
    delete ExpResult;
    return ReturnTensor;
}

void Tensor::SaveToFile(std::string FilePath)
{
    std::ofstream file(FilePath, std::ios::binary);
    SaveToFile(file);
    file.close();
}

void Tensor::SaveToFile(std::ofstream& OpenedFile)
{
    Log::Assert(OpenedFile.is_open(), std::string("This File Is Not Opened."));
    //InitTensor(shape,0)
    //然后是一个size_t的类型，是shape的维度
    //然后把shape数组存进去，全是size_t
    //然后把data数组存进去，全是float

    size_t ProtoDeviceNum = GetDeviceNum();
    ToDevice(0);
    size_t ShapeSize = shape.size();
    OpenedFile.write(reinterpret_cast<const char*>(&ShapeSize), sizeof(ShapeSize));
    OpenedFile.write(reinterpret_cast<const char*>(shape.data()), sizeof(size_t)*shape.size());
    OpenedFile.write(reinterpret_cast<const char*>(GetDevicePointer()), sizeof(float)*ShapeCount);
    ToDevice(ProtoDeviceNum);
}

void Tensor::LoadFromFile(std::ifstream& OpenedFile)
{
    size_t ProtoDeviceNum;
    bool CreateFlag = false;
    if(DPMgr != nullptr)
    {
        ProtoDeviceNum = GetDeviceNum();
        ToDevice(0);
    }
    else
    {
        CreateFlag = true;
    }
    size_t ShapeSize;
    OpenedFile.read(reinterpret_cast<char*>(&ShapeSize), sizeof(ShapeSize));
    std::vector<size_t>LoadShape;
    LoadShape.resize(ShapeSize);
    OpenedFile.read(reinterpret_cast<char*>(LoadShape.data()), sizeof(size_t)*LoadShape.size());
    InitTensor(LoadShape,0);
    OpenedFile.read(reinterpret_cast<char*>(GetDevicePointer()), sizeof(float)*ShapeCount);
    if(!CreateFlag)
    {
        ToDevice(ProtoDeviceNum);
    }
}
void Tensor::LoadFromFile(std::string FilePath)
{
    std::ifstream file(FilePath, std::ios::binary);
    LoadFromFile(file);
    file.close();
}

void Tensor::FillRandomValNormal()
{
    FillRandomValNormal(0.,1.);
}

void Tensor::FillRandomValNormal(float MeanV, float VarianceV)
{
    unsigned Seed = std::chrono::system_clock::now().time_since_epoch().count();
    FillRandomValNormal(MeanV,VarianceV,Seed);
}

void Tensor::FillRandomValNormal(float MeanV, float VarianceV,unsigned Seed)
{
    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        FillRandomValNormalInCPP(GetDevicePointer(), ShapeCount,MeanV, VarianceV, Seed);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().FillRandomValNormalInCPP, 
            ShapeCount,
            {
                GetDeviceBuffer(),
                VBuffer::CVBuffer((int)(ShapeCount)).OpenGLTMPBuffer,
                VBuffer::CVBuffer(MeanV).OpenGLTMPBuffer,
                VBuffer::CVBuffer(VarianceV).OpenGLTMPBuffer,
                VBuffer::CVBuffer(Seed).OpenGLTMPBuffer,
            }
        );
        #endif
    }
    else
    {
        std::default_random_engine Gen(Seed);
        std::normal_distribution<> Dist(MeanV, VarianceV);
        for(size_t a = 0;a<ShapeCount;a++)
        {
            GetDevicePointer()[a] = Dist(Gen);
        }
    }
}

void Tensor::FillRandomValBernoulli(float P)
{
    unsigned Seed = std::chrono::system_clock::now().time_since_epoch().count();
    FillRandomValBernoulli(P,Seed);
}

void Tensor::FillRandomValBernoulli(float P, unsigned Seed)
{
    P = 1-P;
    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        FillRandomValBernoulliInCPP(GetDevicePointer(), ShapeCount,P, Seed);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().FillRandomValBernoulliInCPP, 
            ShapeCount,
            {
                GetDeviceBuffer(),
                VBuffer::CVBuffer((int)(ShapeCount)).OpenGLTMPBuffer,
                VBuffer::CVBuffer(P).OpenGLTMPBuffer,
                VBuffer::CVBuffer(Seed).OpenGLTMPBuffer,
            }
        );
        #endif
    }
    else
    {
        std::default_random_engine Gen(Seed);
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for(size_t a = 0;a<ShapeCount;a++)
        {
            float RandomValue = distribution(Gen);
            GetDevicePointer()[a] = RandomValue > P;
        }
    }
}

void Tensor::FillRandomValUniform()
{
    FillRandomValUniform(0,1);
}

void Tensor::FillRandomValUniform(float MinV, float MaxV)
{
    unsigned Seed = std::chrono::system_clock::now().time_since_epoch().count();
    FillRandomValUniform(MinV, MaxV,Seed);
}

void Tensor::FillRandomValUniform(float MinV, float MaxV, unsigned Seed)
{
    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        FillRandomValUniformInCPP(GetDevicePointer(), ShapeCount,MinV,MaxV, Seed);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().FillRandomValUniformInCPP, 
            ShapeCount,
            {
                GetDeviceBuffer(),
                VBuffer::CVBuffer((int)(ShapeCount)).OpenGLTMPBuffer,
                VBuffer::CVBuffer(MinV).OpenGLTMPBuffer,
                VBuffer::CVBuffer(MaxV).OpenGLTMPBuffer,
                VBuffer::CVBuffer(Seed).OpenGLTMPBuffer,
            }
        );
        #endif
    }
    else
    {
        std::default_random_engine Gen(Seed);
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for(size_t a = 0;a<ShapeCount;a++)
        {
            GetDevicePointer()[a] = distribution(Gen)*(MaxV - MinV) + MinV;
        }
    }
}

Tensor* Tensor::GenerateSignTensor()
{
    Tensor* ReturnTensor = Copy();
    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        GenerateSignTensorInCPP(ReturnTensor->GetDevicePointer(), ReturnTensor->ShapeCount);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().GenerateSignTensorInCPP, 
            ReturnTensor->ShapeCount,
            {
                ReturnTensor->GetDeviceBuffer(),
                VBuffer::CVBuffer((int)(ReturnTensor->ShapeCount)).OpenGLTMPBuffer,
                VBuffer::CVBuffer((float)0).OpenGLTMPBuffer,
            }
        );
        #endif
    }
    else
    {
        for(size_t a = 0;a<ShapeCount;a++)
        {
            if(GetDevicePointer()[a] < 0)ReturnTensor->GetDevicePointer()[a] = 0;
            else ReturnTensor->GetDevicePointer()[a] = 1.;
        }
    }
    return ReturnTensor;
}

Tensor* Tensor::ReLU()
{
    Tensor* SignTensor = GenerateSignTensor();
    Tensor* ReturnTensor = EleMul(SignTensor);
    delete SignTensor;
    return ReturnTensor;
}

Tensor* Tensor::Pow(float Exponent)
{
    Tensor* ReturnTensor = Copy();
    if(GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        PowInCPP(ReturnTensor->GetDevicePointer(), ReturnTensor->ShapeCount, Exponent);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().PowInCPP, 
            ReturnTensor->ShapeCount,
            {
                ReturnTensor->GetDeviceBuffer(),
                VBuffer::CVBuffer((int)(ReturnTensor->ShapeCount)).OpenGLTMPBuffer,
                VBuffer::CVBuffer(Exponent).OpenGLTMPBuffer,
            }
        );
        #endif
    }
    else
    {
        for(size_t a = 0;a<ShapeCount;a++)
        {
            ReturnTensor->GetDevicePointer()[a] = pow(ReturnTensor->GetDevicePointer()[a], Exponent);
        }
    }
    return ReturnTensor;
}

Tensor* Tensor::View(std::vector<size_t> OutputShape, int MinusOneIdx)
{
    Tensor* ReturnTensor = Copy();
    ReturnTensor->shape = OutputShape;
    if(MinusOneIdx >= 0)
    {
        int SpcIdx = 1;
        for(size_t a =0;a<OutputShape.size();a++)
        {
            if(a!=MinusOneIdx)
            {
                SpcIdx*=OutputShape[a];
            }
        }
        ReturnTensor->shape[MinusOneIdx] = ReturnTensor->ShapeCount/SpcIdx;
    }
    return ReturnTensor;
}

Tensor* Tensor::CreateOnehotTensor(std::vector<size_t> InputShape, std::vector<size_t>InputData,size_t TokenLength, size_t DeviceNum)
{
    if(!TokenLength)
    {
        for(size_t a = 0;a<InputData.size();a++)
        {
            TokenLength = std::max(TokenLength, InputData[a]);
        }
        TokenLength += 1;
    }
    InputShape.push_back(TokenLength);
    Tensor* ReturnTensor = new Tensor({InputShape}, DeviceNum);
    ReturnTensor->FillArray(0);
    size_t BaseShape = ReturnTensor->ShapeCount/TokenLength;
    if(ReturnTensor->GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        FillOnehotDataInCPP(ReturnTensor->GetDevicePointer(), BaseShape, TokenLength, InputData.data());
        #endif
        #ifdef OPENGL_USEFUL
        std::vector<int>InputDataUint;
        for(auto& it:InputData)InputDataUint.push_back(it);
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().FillOnehotDataInCPP, 
            BaseShape,
            {
                ReturnTensor->GetDeviceBuffer(),
                VBuffer::CVBuffer((int)(BaseShape)).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(TokenLength)).OpenGLTMPBuffer,
                VBuffer::CVBuffer(InputDataUint.data(), InputDataUint.size()).OpenGLTMPBuffer,
            }
        );
        #endif
    }
    else
    {
        for(size_t a = 0;a<BaseShape;a++)
        {
            ReturnTensor->GetDevicePointer()[a*TokenLength + InputData[a]] = 1.;
        }
    }
    return ReturnTensor;
}

Tensor* Tensor::Sin()
{
    Tensor* ReturnTensor = Copy();
    if(ReturnTensor->GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        TrigonometricFunctionsInCPP(ReturnTensor->GetDevicePointer(), ShapeCount, 0);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().TrigonometricFunctionsInCPP, 
            ShapeCount,
            {
                ReturnTensor->GetDeviceBuffer(),
                VBuffer::CVBuffer((int)(ShapeCount)).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(0)).OpenGLTMPBuffer,
            }
        );
        #endif
    }
    else
    {
        for(size_t a = 0;a < ReturnTensor->ShapeCount;a++)
        {
            ReturnTensor->GetDevicePointer()[a] = std::sin(ReturnTensor->GetDevicePointer()[a]);
        }
    }
    return ReturnTensor;
}
Tensor* Tensor::Cos()
{
    Tensor* ReturnTensor = Copy();
    if(ReturnTensor->GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        TrigonometricFunctionsInCPP(ReturnTensor->GetDevicePointer(), ShapeCount, 1);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().TrigonometricFunctionsInCPP, 
            ShapeCount,
            {
                ReturnTensor->GetDeviceBuffer(),
                VBuffer::CVBuffer((int)(ShapeCount)).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(1)).OpenGLTMPBuffer,
            }
        );
        #endif
    }
    else
    {
        for(size_t a = 0;a < ReturnTensor->ShapeCount;a++)
        {
            ReturnTensor->GetDevicePointer()[a] = std::cos(ReturnTensor->GetDevicePointer()[a]);
        }
    }
    return ReturnTensor;
}

Tensor* Tensor::ArithmeticSequence(std::vector<size_t> InputShape, float A1, float Arithmetic, size_t InputDeviceNum)
{
    Tensor* ReturnTensor = new Tensor(InputShape, InputDeviceNum);
    if(ReturnTensor->GetDeviceNum())
    {
        CudaDimVec InputShapeArray = ReturnTensor->TransformFromStdVector(InputShape, InputShape.size());
        #ifdef CUDA_USEFUL
        ArithmeticSequenceInCPP(ReturnTensor->GetDevicePointer(), InputShapeArray.Shape, InputShape.size(), A1, Arithmetic);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().ArithmeticSequenceInCPP, 
            ReturnTensor->ShapeCount,
            {
                ReturnTensor->GetDeviceBuffer(),
                VBuffer::CVBuffer(InputShapeArray.ToInt(), InputShapeArray.ShapeLen).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(InputShape.size())).OpenGLTMPBuffer,
                VBuffer::CVBuffer(A1).OpenGLTMPBuffer,
                VBuffer::CVBuffer(Arithmetic).OpenGLTMPBuffer,
            }
        );
        #endif
    }
    else
    {
        size_t CurIndex;
        for(size_t a = 0;a<ReturnTensor->ShapeCount;a++)
        {
            CurIndex = a%InputShape[InputShape.size()-1];
            ReturnTensor->GetDevicePointer()[a] = CurIndex*Arithmetic + A1;
        }
    }
    return ReturnTensor;
}

Tensor* Tensor::PositionalEncoding(int DModel, int MaxLen, size_t InputDeviceNum)
{
    Tensor* Position = Tensor::ArithmeticSequence({ MaxLen * 1U }, 0, 1, InputDeviceNum);
    Position->shape.push_back(1);
    Tensor* DivTermBase = Tensor::ArithmeticSequence({ (DModel+1)/2 * 1U }, 0, 2, InputDeviceNum);
    float DivTermNum = -std::log(10000.0) / DModel;
    Tensor* DivTermBaseMulNum = DivTermBase->MulScalar(DivTermNum);
    delete DivTermBase;
    Tensor* DivTerm = DivTermBaseMulNum->EleExp(M_E);
    DivTerm->shape = { 1,DivTerm->ShapeCount };
    delete DivTermBaseMulNum;
    Tensor* PosMulDiv = Position->Matmul(DivTerm);
    delete DivTerm;
    delete Position;
    Tensor* CosDivTerm = PosMulDiv->Cos();
    Tensor* SinDivTerm = PosMulDiv->Sin();
    delete PosMulDiv;
    auto ProtoShape = CosDivTerm->shape;
    ProtoShape.push_back(ProtoShape[ProtoShape.size() - 1] * 2);
    ProtoShape[ProtoShape.size() - 2] = 1;
    CosDivTerm->shape = { CosDivTerm->ShapeCount, 1 };
    SinDivTerm->shape = { SinDivTerm->ShapeCount, 1 };
    Tensor* CombDivTerm = SinDivTerm->TensorSplice(CosDivTerm, 1);
    delete CosDivTerm;
    delete SinDivTerm;
    CombDivTerm->shape = ProtoShape;
    return CombDivTerm;
}   

std::vector<Tensor*> Tensor::GenerateSplitTensor(int SplitSize, int Dim)
{
    std::vector<int>SplitSections;
    int ProtoDimSize = shape[Dim];
    for(size_t a = 0; a < SplitSize ;a++)
    {
        if(a < (SplitSize-ProtoDimSize%SplitSize))SplitSections.push_back(ProtoDimSize/SplitSize);
        else SplitSections.push_back(ProtoDimSize/SplitSize + 1);
    }
    return GenerateSplitTensor(SplitSections, Dim);
}
std::vector<Tensor*> Tensor::GenerateSplitTensor(std::vector<int> SplitSections, int Dim)
{
    std::vector<Tensor*>ReturnVec;
    size_t FirstDim = 1;
    size_t SecondDim = 1;
    for (size_t a = 0; a < Dim; a++)FirstDim *= shape[a];
    for(size_t a = Dim;a < shape.size();a++)SecondDim*=shape[a];
    std::vector<size_t>CurDim;
    size_t AllParts = 0;
    for (size_t a = 0; a < SplitSections.size(); a++)AllParts += SplitSections[a];
    for (size_t a = 0; a < SplitSections.size(); a++)CurDim.push_back((SecondDim * SplitSections[a]) / AllParts);
    size_t LastSum = 0;
    for (size_t a = 0; a < SplitSections.size(); a++)
    {
        Tensor* ThisTMPRes = GetUnitTensor({ CurDim[a],CurDim[a] }, GetDeviceNum());
        if (LastSum != 0)
        {
            Tensor* ZeroPre = new Tensor({ LastSum,CurDim[a]}, GetDeviceNum());
            ZeroPre->FillArray(0);
            Tensor* PreCurTensor = ZeroPre->TensorSplice(ThisTMPRes, 0);
            delete ThisTMPRes;
            delete ZeroPre;
            ThisTMPRes = PreCurTensor;
        }
        if (LastSum + CurDim[a] < SecondDim)
        {
            Tensor* ZeroLast = new Tensor({ SecondDim - CurDim[a] - LastSum,CurDim[a] }, GetDeviceNum());
            ZeroLast->FillArray(0);
            Tensor* LastCurTensor = ThisTMPRes->TensorSplice(ZeroLast,0);
            delete ThisTMPRes;
            delete ZeroLast;
            ThisTMPRes = LastCurTensor;
        }
        LastSum += CurDim[a];
        ReturnVec.push_back(ThisTMPRes);
    }
    return ReturnVec;
}

std::vector<Tensor*> Tensor::TensorSplit(int SplitSize, int Dim)
{
    std::vector<int>SplitSections;
    int ProtoDimSize = shape[Dim];
    for (size_t a = 0; a < SplitSize; a++)
    {
        if (a < (SplitSize - ProtoDimSize % SplitSize))SplitSections.push_back(ProtoDimSize / SplitSize);
        else SplitSections.push_back(ProtoDimSize / SplitSize + 1);
    }
    return TensorSplit(SplitSections, Dim);
}
std::vector<Tensor*> Tensor::TensorSplit(std::vector<int> SplitSections, int Dim)
{
    auto GenLeftMul = GenerateSplitTensor(SplitSections, Dim);
    std::vector<Tensor*>Res;
    size_t PreDims = 1;
    size_t LastDims = 1;
    for (size_t a = 0; a < shape.size(); a++)
    {
        if (a < Dim)PreDims *= shape[a];
        else LastDims *= shape[a];
    }
    Tensor* ViewTensor = View({ PreDims,LastDims });
    for (size_t a = 0; a < GenLeftMul.size(); a++)
    {
        Tensor* ResTMPTensor = ViewTensor->Matmul(GenLeftMul[a]); 
        auto ReturnShape = shape;
        ReturnShape[Dim] = SplitSections[a];
        Res.push_back(ResTMPTensor->View(ReturnShape));
        delete ResTMPTensor;
    }
    delete ViewTensor;
    for (size_t a = 0; a < GenLeftMul.size(); a++)delete GenLeftMul[a];
    return Res;
}

std::vector<Tensor*> Tensor::GenerateCatTensor(std::vector<Tensor*>InputTensors, int Dim)
{
    std::vector<Tensor*>Res;
    int TargetDim = 0;
    for (size_t a = 0; a < InputTensors.size(); a++)TargetDim += InputTensors[a]->shape[Dim];
    for (size_t a = 0; a < InputTensors[0]->shape.size(); a++)
    {
        if(a>Dim)TargetDim*=InputTensors[0]->shape[a];
    }
    size_t PreDim = 0;
    for (size_t a = 0; a < InputTensors.size(); a++)
    {
        size_t FirstDim = 1;
        size_t LastDim = 1;
        for (size_t b = 0; b < InputTensors[a]->shape.size(); b++)
        {
            if (b < Dim)FirstDim *= InputTensors[a]->shape[b];
            else LastDim *= InputTensors[a]->shape[b];
        }
        Tensor* ThisTMPRes = GetUnitTensor({ LastDim, LastDim }, InputTensors[a]->GetDeviceNum());
        if (PreDim != 0)
        {
            Tensor* ZeroPre = new Tensor({ LastDim,PreDim }, InputTensors[a]->GetDeviceNum());
            ZeroPre->FillArray(0);
            Tensor* PreCurTensor = ZeroPre->TensorSplice(ThisTMPRes, 1);
            delete ThisTMPRes;
            delete ZeroPre;
            ThisTMPRes = PreCurTensor;
        }
        if (PreDim + LastDim < TargetDim)
        {
            Tensor* ZeroPre = new Tensor({ LastDim,TargetDim-PreDim- LastDim }, InputTensors[a]->GetDeviceNum());
            ZeroPre->FillArray(0);
            Tensor* PreCurTensor = ThisTMPRes->TensorSplice(ZeroPre, 1);
            delete ThisTMPRes;
            delete ZeroPre;
            ThisTMPRes = PreCurTensor;
        }
        Res.push_back(ThisTMPRes);
        PreDim += LastDim;
    }
    return Res;
}

Tensor* Tensor::TensorCat(std::vector<Tensor*>InputTensors, int Dim)
{
    auto GenRightMul = GenerateCatTensor(InputTensors, Dim);
    int TargetDim = 0;
    for (size_t a = 0; a < InputTensors.size(); a++)TargetDim += InputTensors[a]->shape[Dim];
    Tensor* Res = nullptr;
    for (size_t a = 0; a < InputTensors.size(); a++)
    {
        size_t FirstDim = 1;
        size_t LastDim = 1;
        for (size_t b = 0; b < InputTensors[a]->shape.size(); b++)
        {
            if (b < Dim)FirstDim *= InputTensors[a]->shape[b];
            else LastDim *= InputTensors[a]->shape[b];
        }
        Tensor* ViewTensor = InputTensors[a]->View({ FirstDim, LastDim });
        Tensor* ThisResTensor = ViewTensor->Matmul(GenRightMul[a]);
        delete ViewTensor;
        if (Res == nullptr)Res = ThisResTensor;
        else
        {
            Tensor* TMPTensor = Res->Add(ThisResTensor);
            delete ThisResTensor;
            delete Res;
            Res = TMPTensor;
        }
    }
    for (size_t a = 0; a < GenRightMul.size(); a++)delete GenRightMul[a];
    std::vector<size_t>TrueShape = InputTensors[0]->shape;
    TrueShape[Dim] = TargetDim;
    Tensor* TMPRes = Res->View(TrueShape);
    delete Res;
    return TMPRes;
}

Tensor* Tensor::GenerateTrilOnes(std::vector<size_t> InputShape, int Diagonal, size_t DeviceNum)
{
    //保留下三角
    Tensor* ReturnTensor = new Tensor(InputShape, DeviceNum);
    size_t ShapeLen = ReturnTensor->shape.size();
    if(ReturnTensor->GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        GenerateTrilOnesInCPP(ReturnTensor->GetDevicePointer(), ReturnTensor->ShapeCount, ReturnTensor->shape[ShapeLen-2], ReturnTensor->shape[ShapeLen-1], Diagonal);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().GenerateTrilOnesInCPP, 
            ReturnTensor->ShapeCount,
            {
                ReturnTensor->GetDeviceBuffer(),
                VBuffer::CVBuffer((int)(ReturnTensor->ShapeCount)).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(ReturnTensor->shape[ShapeLen-2])).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(ReturnTensor->shape[ShapeLen-1])).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(Diagonal)).OpenGLTMPBuffer,
            }
        );
        #endif
    }
    else
    {

        size_t Row = ReturnTensor->shape[ShapeLen-2];
        size_t Col = ReturnTensor->shape[ShapeLen-1];
        for(size_t Index = 0;Index < ReturnTensor->ShapeCount;Index++)
        {
            size_t TrueIndex = Index%(Row*Col);
            int ThisRow = TrueIndex/Col;
            int ThisCol = TrueIndex%Col;
            if(ThisCol <= ThisRow + Diagonal)ReturnTensor->GetDevicePointer()[Index] = 1;
            else ReturnTensor->GetDevicePointer()[Index] = 0;
        }
    }
    return ReturnTensor;
}

Tensor* Tensor::Tril(int Diagonal)
{
    Tensor* TrilOnes = GenerateTrilOnes(shape, Diagonal, GetDeviceNum());
    Tensor* ResTensor = EleMul(TrilOnes);
    delete TrilOnes;
    return ResTensor;
}

Tensor* Tensor::Transpose(int FirstDim, int SecondDim)
{
    auto ReturnShape = shape;
    if(FirstDim < 0)FirstDim = shape[shape.size()-FirstDim];
    if(SecondDim < 0)SecondDim = shape[shape.size()-SecondDim];
    ReturnShape[FirstDim] = shape[SecondDim];
    ReturnShape[SecondDim] = shape[FirstDim];
    Tensor* ReturnTensor = View(ReturnShape);
    CudaDimVec OutputShapeArray = TransformFromStdVector(ReturnTensor->shape, ReturnTensor->shape.size());
    size_t* OutputShapePointer = OutputShapeArray.Shape;
    if(ReturnTensor->GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        TransposeInCPP(ReturnTensor->GetDevicePointer(), GetDevicePointer(), OutputShapePointer, shape.size(), FirstDim, SecondDim);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().TransposeInCPP, 
            ReturnTensor->ShapeCount,
            {
                ReturnTensor->GetDeviceBuffer(),
                GetDeviceBuffer(),
                VBuffer::CVBuffer(OutputShapeArray.ToInt(),OutputShapeArray.ShapeLen).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(shape.size())).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(FirstDim)).OpenGLTMPBuffer,
                VBuffer::CVBuffer((int)(SecondDim)).OpenGLTMPBuffer,
            }
        );
        #endif
    }
    else
    {
        int LeftDim = std::min(FirstDim, SecondDim);//最左边的维度
        int RightDim = std::max(FirstDim, SecondDim);//最右边的维度
        int MidRightElement = 1;//所有的要变换的维度的元素个数
        int RightElement = 1;//比要变换的右边的维度还小的
        int MidElement = 1;//中间的维度
        int OutputShapeSize = shape.size();
        int OutputShapeCount = ShapeCount;
        size_t* OutputShape = OutputShapePointer;
        auto OutputData = ReturnTensor->GetDevicePointer();
        auto InputData = GetDevicePointer();
        for(size_t a=0;a<OutputShapeSize;a++)
        {
          if(a < LeftDim)continue;
          MidRightElement*=OutputShape[a];
          if(a > RightDim)RightElement*=OutputShape[a];
          if(a < RightDim&&a > LeftDim)MidElement*=OutputShape[a];
        }
        for(int Index = 0;Index < OutputShapeCount;Index++)
        {
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
    }
    return ReturnTensor;
}

Tensor* Tensor::EleLog()
{
    Tensor* ReturnTensor = Copy();
    if(ReturnTensor->GetDeviceNum())
    {
        #ifdef CUDA_USEFUL
        EleLogInCPP(ReturnTensor->GetDevicePointer(), ShapeCount);
        #endif
        #ifdef OPENGL_USEFUL
        GPUDeviceProcess::I().ProcessGLSLFun
        (
            GLSL::I().EleLogInCPP, 
            ShapeCount,
            {
                ReturnTensor->GetDeviceBuffer(),
                VBuffer::CVBuffer((int)(ShapeCount)).OpenGLTMPBuffer,
            }
        );
        #endif
    }
    else
    {
        for(size_t Index = 0;Index < ShapeCount; Index++)
        {
            ReturnTensor->GetDevicePointer()[Index] = std::log(GetDevicePointer()[Index]);
        }
    }
    return ReturnTensor;
}