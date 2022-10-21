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
    cudaMallocInCPP(&DataGPU, ShapeCount, DeviceNum);
    DataToGPU(DataCPU, DataGPU, ShapeCount);
    free(DataCPU);
}

void Tensor::FillArray(float Scalar)
{
    if(Device == "GPU")FillArrayInCPP(DataGPU, Scalar, ShapeCount);
    else for(int a=0;a<ShapeCount;a++)DataCPU[a] = Scalar;
}

Tensor* Tensor::AddArray(Tensor* Input)
{
    Tensor* Output = new Tensor(shape, Device, DeviceNum);
    if(Device == "GPU")AddArrayInCPP(Output->DataGPU, DataGPU, Input->DataGPU, ShapeCount);
    else for(int a=0;a<ShapeCount;a++)Output->DataCPU[a] = DataCPU[a] + Input->DataCPU[a];
    return Output;
}