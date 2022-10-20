#include "Tensor.h"
Tensor::Tensor(std::vector<size_t>shape, std::string Device)
{
    this->shape = shape;
    this->Device = Device;
    for(int a=0;a<shape.size();a++)ShapeCount*=shape[a];
    if(Device == "GPU")cudaMallocInCPP(&(this->DataGPU), ShapeCount);
    else this->DataCPU = (float*)malloc(sizeof(float)*ShapeCount);
}

void Tensor::PrintData()
{
    if(Device == "GPU")
    {
        ToCPU();
        for(int a=0;a<ShapeCount;a++)std::cout<<DataCPU[a]<<" ";
        std::cout<<std::endl;
        ToGPU();
    }
    else
    {
        for(int a=0;a<ShapeCount;a++)std::cout<<DataCPU[a]<<" ";
        std::cout<<std::endl;
    }
}

void Tensor::ToCPU()
{
    if(Device == "CPU")return;
    this->DataCPU = (float*)malloc(sizeof(float)*ShapeCount);
    DataToCPU(DataCPU, DataGPU, ShapeCount);
    cudaFreeInCPP(DataGPU);
}

void Tensor::ToGPU()
{
    if(Device == "GPU")return;
    cudaMallocInCPP(&DataGPU, ShapeCount);
    DataToGPU(DataCPU, DataGPU, ShapeCount);
    free(DataCPU);
}

void Tensor::FillArray(float Scalar)
{
    FillArrayInCPP(DataGPU, Scalar, ShapeCount);
}


Tensor* Tensor::AddArray(Tensor* Input)
{
    Tensor* Output = new Tensor(shape, Device);
    AddArrayInCPP(Output->DataGPU, DataGPU, Input->DataGPU, ShapeCount);
    return Output;
}