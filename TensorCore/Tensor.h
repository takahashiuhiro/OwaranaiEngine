#include "TensorInclude.h"

extern "C" void AddArray(float* Output, float* InputFirst, float* InputSecond, size_t Size);
extern "C" void cudaMallocInCPP(float** Input, size_t Size);
extern "C" void DataToGPU(float* CPUPointer, float* GPUPointer, size_t Size);
extern "C" void DataToCPU(float* CPUPointer, float* GPUPointer, size_t Size);
extern "C" void cudaFreeInCPP(float* Input);
extern "C" void AddArrayInCPP(float* Output, float* InputFirst, float* InputSecond, size_t Size);
extern "C" void FillArrayInCPP(float* Input, float Scalar,size_t Size);

struct Tensor
{
public:
    Tensor(){}
    Tensor(std::vector<size_t>shape, std::string Device);

    std::vector<size_t>shape;
    size_t ShapeCount = 1;
    float* DataCPU;
    float* DataGPU;
    std::string Device = "CPU";
    int DeviceNum = 0;

    void ToGPU();
    void ToCPU();
    void PrintData();
    void FillArray(float Scalar);
    Tensor* AddArray(Tensor* Input);
};