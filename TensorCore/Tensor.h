#include "TensorInclude.h"

extern "C" void CudaMallocInCpp(float* Input, size_t Size);


struct Tensor
{
public:
    Tensor(){}
    Tensor(std::vector<size_t>shape, int Device);

    std::vector<size_t>shape;
    float* Data;
    int Device = 0;
};