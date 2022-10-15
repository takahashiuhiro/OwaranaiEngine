#include "Tensor.h"


Tensor::Tensor(std::vector<size_t>shape, int Device)
{
    this->shape = shape;
    this->Device = Device;
    size_t ShapeCount = 1;
    for(int a=0;a<shape.size();a++)ShapeCount*=shape[a];
    CudaMallocInCpp(this->Data, ShapeCount);
    std::cout<<"123123"<<std::endl;
}