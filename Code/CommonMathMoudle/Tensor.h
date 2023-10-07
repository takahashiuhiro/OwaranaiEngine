#pragma once
#include <memory>
#include "../CommonDataStructure/Log.h"
#include "MathHelpers.h"
#include <cmath>
#include <fstream>
#include <random>
#include <chrono>

#ifdef CUDA_USEFUL
extern "C" void cudaMallocInCPP(float** Input, size_t Size, size_t DeviceNum);
extern "C" void DataCPUToGPU(float* CPUPointer, float* GPUPointer, size_t Size);
extern "C" void DataGPUToCPU(float* CPUPointer, float* GPUPointer, size_t Size);
extern "C" void DataGPUToGPU(float* GPUPointerOutput, float* GPUPointerInput, size_t Size);
extern "C" void cudaFreeInCPP(float* Input);
extern "C" void AddArrayInCPP(float* Output, float* InputFirst, float* InputSecond, size_t Size);
extern "C" void FillArrayInCPP(float* Input, float Scalar,size_t Size);
extern "C" void AddInCPP(float* Output, float* HighDimInput, size_t HighDimSize, float* LowDimInput, size_t LowDimSize);
extern "C" void EleMulInCPP(float* Output, float* HighDimInput, size_t HighDimSize, float* LowDimInput, size_t LowDimSize);
extern "C" void MulScalarInCPP(float* Output,float* Input, float Scalar,size_t Size);
extern "C" void AddScalarInCPP(float* Output,float* Input, float Scalar,size_t Size);
extern "C" void DotArrayInCPP(float* Output, float* InputFirst, float* InputSecond, size_t Size);
extern "C" void MatmulInCPP
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
);
extern "C" void TInCPP(float* Output, float* Input, size_t *MatrixShape, size_t ShapeCount);
extern "C" void SumTensorDimInCPP(float* OutputData, float* InputData, size_t *InputShape, size_t InputShapeLen, size_t InputDim, size_t OutputShapeCount);
extern "C" void MaximumOrMinimumTensorDimInCPP(float* OutputData, float* InputData, size_t *InputShape, size_t InputShapeLen, size_t InputDim, size_t OutputShapeCount, bool IsMaximum);
extern "C" void TensorSpliceInCPP(float* OutputData, float* InputDataFirst, float* InputDataSecond, size_t* InputShapeFirst, size_t* InputShapeSecond, size_t InputShapeLen, size_t InputDim, size_t OutputShapeCount);
extern "C" void GetUnitTensorInCPP(float* OutputData, size_t* InputShape, size_t OutputShapeCount, size_t InputShapeLen);
extern "C" void GaussianEliminationInCPP(float* OutputData, size_t BatchSize, size_t Row, size_t Column);
extern "C" void GetTensorBy2ShapeVectorInCPP(float* OutputData, float* InputData, size_t* InputShape,size_t* OutputShape,size_t* StartShape, size_t* EndShape, size_t ShapeLen);
extern "C" void EleExpInCPP(float* OutputData, size_t OutputShape, float BaseNum);
extern "C" void EleInverseInCPP(float* OutputData, size_t OutputShape);
extern "C" void BroadCastToInCPP(float* OutputData, float* InputData, size_t* OutputShape, size_t* InputShape, size_t ShapeLen, size_t OutputShapeCount);
extern "C" void FillRandomValNormalInCPP(float* OutputData, size_t OutputShapeCount, unsigned Seed);
#endif

struct CudaDimVec
{
  size_t Shape[8];
};

static size_t DeviceNumToCuda(size_t DeviceNum)
{
    //0是cpu
    return DeviceNum-1;
}

struct DevicePointerManager
{

    DevicePointerManager(size_t DeviceNum, size_t ShapeCount)
    {
        this->MaxDeviceNum = 2;
        InitDevicePointerManager(DeviceNum, ShapeCount);
    }

    DevicePointerManager(size_t MaxDeviceNum, size_t DeviceNum, size_t ShapeCount)
    {
        this->MaxDeviceNum = MaxDeviceNum;
        InitDevicePointerManager(DeviceNum, ShapeCount);
    }

    void InitDevicePointerManager(size_t DeviceNum, size_t ShapeCount)
    {
        for(size_t a=0;a<MaxDeviceNum;a++)
        {
            DataPointers.push_back(nullptr);
        }
        SetDevice(DeviceNum, ShapeCount);
    }

    ~DevicePointerManager()
    {
        FreeOldDevice(this->DeviceNum);
    }

    std::vector<float*>DataPointers;
    /**最大的设备数.*/
    size_t MaxDeviceNum = 1;
    /**当前的设备数.*/
    size_t DeviceNum = ImpossibleFrameMaxDeviceNum();

    size_t FrameMaxDeviceNum()
    {
        return 5000;
    }

    size_t ImpossibleFrameMaxDeviceNum()
    {
        return FrameMaxDeviceNum()+1;
    }

    float* GetDevicePointer()
    {
        return DataPointers[DeviceNum];
    }

    void FreeOldDevice(size_t OldDeviceNum)
    {
        //释放对应设备的内存
        if(OldDeviceNum == ImpossibleFrameMaxDeviceNum())return;
        if(!DataPointers[OldDeviceNum])return;
        if(!OldDeviceNum)
        {
            free(DataPointers[OldDeviceNum]);
        }
        else
        {
            #ifdef CUDA_USEFUL
            cudaFreeInCPP(DataPointers[OldDeviceNum]);
            #endif
        }
    }

    void SetDevice(size_t NewDeviceNum, size_t ShapeCount)
    {
        //设置对应设备的内存
        size_t OldDeviceNum = DeviceNum;
        if(NewDeviceNum == OldDeviceNum)return;
        if(!NewDeviceNum)
        {
            DataPointers[NewDeviceNum] = (float*)malloc(sizeof(float)*ShapeCount);
            #ifdef CUDA_USEFUL
            if(OldDeviceNum !=ImpossibleFrameMaxDeviceNum())DataGPUToCPU(DataPointers[NewDeviceNum], DataPointers[OldDeviceNum], ShapeCount);
            #endif
        }
        else
        {
            bool CudaFlag = 0;
            #ifdef CUDA_USEFUL
            CudaFlag = 1;
            cudaMallocInCPP(&DataPointers[NewDeviceNum], ShapeCount, DeviceNumToCuda(NewDeviceNum));
            #endif
            Log::Assert(CudaFlag, std::string("Use Cuda Branch But..."));
            if(OldDeviceNum !=ImpossibleFrameMaxDeviceNum())
            {
                #ifdef CUDA_USEFUL
                if(!OldDeviceNum)
                {
                    DataCPUToGPU(DataPointers[OldDeviceNum], DataPointers[NewDeviceNum], ShapeCount);
                }
                else
                {
                    DataGPUToGPU(DataPointers[NewDeviceNum], DataPointers[OldDeviceNum], ShapeCount);
                }
                #endif
            }
        }
        FreeOldDevice(OldDeviceNum);
        this->DeviceNum = NewDeviceNum;
    }
};

class Tensor
{
public:
    Tensor(){}
    Tensor(std::vector<size_t>shape);
    Tensor(std::vector<size_t>shape, size_t DeviceNum);

    Tensor* CopyNewEmptyTensor();
    Tensor* Copy();
    static Tensor* CreateTensorByLoadPath(std::ifstream& OpenedFile, size_t DeviceNum);
    static Tensor* CreateTensorByLoadPath(std::ifstream& OpenedFile);
    static Tensor* CreateTensorByLoadPath(std::string LoadPath, size_t DeviceNum);
    static Tensor* CreateTensorByLoadPath(std::string LoadPath);

    void InitTensor(std::vector<size_t>shape, size_t DeviceNum);
    ~Tensor(){};

    /**数据管理.*/
    std::shared_ptr<DevicePointerManager> DPMgr = nullptr;
    /**shape of the tensor*/
    std::vector<size_t>shape;
    /**all data num*/
    size_t ShapeCount = 1;

    void ToDevice(size_t NewDeviceNum)
    {
        DPMgr->SetDevice(NewDeviceNum, ShapeCount);
    }

    size_t GetDeviceNum()
    {
        return DPMgr->DeviceNum;
    }

    float* GetDevicePointer()
    {
        return DPMgr->GetDevicePointer();
    }
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
    /**return a array from shape vector*/
    CudaDimVec TransformFromStdVector(std::vector<size_t> InputVector, size_t ShapeLen);
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
    /**make the tensor Transpose*/
    Tensor* T();
    /**对标定维度求Sum.*/
    Tensor* Sum(std::vector<size_t>InputDims);
    /**Get a sum tensor by specifying dimensions*/
    Tensor* SumTensorDim(size_t InputDim);
    /**Get a average tensor by specifying dimensions*/
    Tensor* AverageTensorDim(size_t InputDim);
    /**高斯消元.*/
    void GaussianElimination();
    /**张量按指定维度拼接.*/
    Tensor* TensorSplice(Tensor* InputTensor, int SpliceDim);
    /**返回一个常量单位矩阵.*/
    static Tensor* GetUnitTensor(std::vector<size_t>ReturnShape, size_t ReturnDeviceNum);
    /**在矩阵中通过两个输入维度扣出新的矩阵.*/
    Tensor* GetTensorBy2ShapeVector(std::vector<size_t>StartShape, std::vector<size_t>EndShape);
    /**矩阵求逆.*/
    Tensor* Inverse();
    /**元素上的逆.*/
    Tensor* EleInverse();
    /**求最大值或者最小值.*/
    Tensor* MaximumOrMinimum(size_t InputDim,  bool IsMaximum);
    Tensor* Maximum(size_t InputDim);
    Tensor* Minimum(size_t InputDim);
    /**对元素求指数函数.*/
    Tensor* EleExp(float BaseNum);
    /**把矩阵广播到输入形状.*/
    Tensor* BroadCastTo(std::vector<size_t>BroadCastShape);
    /**是否能广播.*/
    bool CanBroadCastTo(std::vector<size_t>BroadCastShape);
    /**softmax.*/
    Tensor* Softmax(size_t InputDim);
    /**把张量存成二进制文件.*/
    void SaveToFile(std::ofstream& OpenedFile);
    void SaveToFile(std::string FilePath);
    /**从二进制文件里取出张量.*/
    void LoadFromFile(std::ifstream& OpenedFile);
    void LoadFromFile(std::string FilePath);
    /**填充随机数.*/
    void FillRandomValNormal();
    void FillRandomValNormal(unsigned Seed);
};