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
    if(Device == "GPU")cudaMallocInCPP(&(this->DataGPU), ShapeCount, DeviceNum);
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
        GPUflag = 1;
        ToCPU();
    }
    for(int a=0;a<ShapeCount;a++)
    {
        std::cout<<DataCPU[a]<<" ";
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
    if(Device == "CPU")return;
    this->DataCPU = (float*)malloc(sizeof(float)*ShapeCount);
    DataToCPU(DataCPU, DataGPU, ShapeCount);
    cudaFreeInCPP(DataGPU);
    Device = "CPU";
}

void Tensor::ToGPU()
{
    if(Device == "GPU")return;
    cudaMallocInCPP(&DataGPU, ShapeCount, DeviceNum);
    DataToGPU(DataCPU, DataGPU, ShapeCount);
    free(DataCPU);
    Device = "GPU";
}

void Tensor::FillArray(float Scalar)
{
    if(Device == "GPU")FillArrayInCPP(DataGPU, Scalar, ShapeCount);
    else for(int a=0;a<ShapeCount;a++)DataCPU[a] = Scalar;
}

Tensor* Tensor::AddScalar(float Scalar)
{
    Tensor* Output = new Tensor(shape, Device, DeviceNum);
    if(Device == "GPU")AddScalarInCPP(Output->DataGPU, DataGPU, Scalar, ShapeCount);
    else for(int a=0;a<ShapeCount;a++)Output->DataCPU[a] = DataCPU[a]+ Scalar;
    return Output;
}

Tensor* Tensor::MulScalar(float Scalar)
{
    Tensor* Output = new Tensor(shape, Device, DeviceNum);
    if(Device == "GPU")MulScalarInCPP(Output->DataGPU, DataGPU, Scalar, ShapeCount);
    else for(int a=0;a<ShapeCount;a++)Output->DataCPU[a] = DataCPU[a] *Scalar;
    return Output;
}

Tensor* Tensor::AddArray(Tensor* Input)
{
    Tensor* Output = new Tensor(shape, Device, DeviceNum);
    if(Device == "GPU")AddArrayInCPP(Output->DataGPU, DataGPU, Input->DataGPU, ShapeCount);
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
        AddInCPP(Output->DataGPU, HighDimTensor->DataGPU, HighDimTensor->ShapeCount, LowDimTensor->DataGPU, LowDimTensor->ShapeCount);
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
        EleMulInCPP(Output->DataGPU, HighDimTensor->DataGPU, HighDimTensor->ShapeCount, LowDimTensor->DataGPU, LowDimTensor->ShapeCount);
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
        GPUflag = 1;
        ToCPU();
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
        GPUflag = 1;
        ToCPU();
    }
    DataCPU[GetIndex(FindIndex)] = Value;
    if(GPUflag)ToGPU();
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
                Output->ToGPU();
                DotArrayInCPP(Output->DataGPU, DataGPU, Input->DataGPU, ShapeCount);
            }
            else for(int a=0;a<ShapeCount;a++)Output->DataCPU[0] += DataCPU[a]*Input->DataCPU[a];
        }
        else
        {
            
        }
    }
    else
    {
        if(Input->shape.size() == 1) 
        {

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
            OutputShape.push_back(shape[shape.size()-2]);
            OutputShape.push_back(Input->shape[Input->shape.size()-1]);
            Output = new Tensor(OutputShape, Device, DeviceNum);
            if(Device == "GPU")
            {
                std::cout<<"Shape Test"<<std::endl;
                for(int a=0;a<OutputShape.size();a++)
                {
                    std::cout<<OutputShape[a]<<" ";
                }
                std::cout<<std::endl;
            }
        }
    }
    return Output;
}