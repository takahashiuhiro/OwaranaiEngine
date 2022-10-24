#include "TensorInclude.h"

extern "C" void cudaMallocInCPP(float** Input, size_t Size, size_t DeviceNum);
extern "C" void DataToGPU(float* CPUPointer, float* GPUPointer, size_t Size);
extern "C" void DataToCPU(float* CPUPointer, float* GPUPointer, size_t Size);
extern "C" void cudaFreeInCPP(float* Input);
extern "C" void AddArrayInCPP(float* Output, float* InputFirst, float* InputSecond, size_t Size);
extern "C" void FillArrayInCPP(float* Input, float Scalar,size_t Size);
extern "C" void AddInCPP(float* Output, float* HighDimInput, size_t HighDimSize, float* LowDimInput, size_t LowDimSize);
extern "C" void EleMulInCPP(float* Output, float* HighDimInput, size_t HighDimSize, float* LowDimInput, size_t LowDimSize);
extern "C" void MulScalarInCPP(float* Output,float* Input, float Scalar,size_t Size);
extern "C" void AddScalarInCPP(float* Output,float* Input, float Scalar,size_t Size);

struct Tensor
{
public:
    Tensor(){}
    Tensor(std::vector<size_t>shape);
    Tensor(std::vector<size_t>shape, std::string Device, size_t DeviceNum);

    /**shape of the tensor*/
    std::vector<size_t>shape;
    /**all data num*/
    size_t ShapeCount = 1;
    float* DataCPU;
    float* DataGPU;
    std::string Device = "CPU";
    size_t DeviceNum = 0;
    /**move data to gpu*/
    void ToGPU();
    /**move data to cpu*/
    void ToCPU();
    /**print data of this tensor*/
    void PrintData();
    /**make a scalar fill array of the tensor*/
    void FillArray(float Scalar);
    /**get index of a vector like std::vector<size_t>{a,b,..}*/
    size_t GetIndex(std::vector<size_t> FindIndex);
    /**get vector from data index*/
    std::vector<size_t> GetDim(size_t DataIndex);
    /**get value tensor[a][b][..]..  and you should input an vector like std::vector<size_t>{a,b,..}*/
    float GetV(std::vector<size_t> FindIndex);
    /**set value tensor[a][b][..]..  and you should input an vector like std::vector<size_t>{a,b,..}*/
    void SetV(std::vector<size_t> FindIndex, float Value);
    /**make two tensor add their array and they must be same shapecount*/
    Tensor* AddArray(Tensor* Input);
    /**add two tensor*/
    Tensor* Add(Tensor* Input);
    /**mul two tensor across elements*/
    Tensor* EleMul(Tensor* Input);
    /**add a scalar*/
    Tensor* AddScalar(float Scalar);
    /**mul a scalar*/
    Tensor* MulScalar(float Scalar);
    /**tensor matmul*/
    Tensor* Matmul(Tensor* Input);
};