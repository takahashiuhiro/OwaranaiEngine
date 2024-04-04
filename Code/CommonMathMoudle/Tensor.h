#pragma once
#include "../ForPlatforms.h"
#include <memory>
#include "../CommonDataStructure/Log.h"
#include "MathHelpers.h"
#include <cmath>
#include <fstream>
#include <random>
#include <chrono>
#include "DevicePointerManager.h"

class Tensor
{
public:
    Tensor(){}
    Tensor(std::vector<size_t>shape);
    Tensor(std::vector<size_t>shape, size_t DeviceNum);
    /**小的拷贝用上面这个，大的只读用下面那个，不能是引用，否则只会走上面这个.*/
    Tensor(std::vector<size_t>shape, size_t DeviceNum, std::vector<float> InputData);
    Tensor(std::vector<size_t>shape, size_t DeviceNum, std::vector<float>* InputData);

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
    /**把张量挪到其他设备上.*/
    void ToDevice(size_t NewDeviceNum){DPMgr->SetDevice(NewDeviceNum, ShapeCount);}
    /**获取张量设备.*/
    size_t GetDeviceNum(){return DPMgr->DeviceNum;}
    /**获取张量数据指针.*/
    float* GetDevicePointer(){return DPMgr->GetDevicePointer();}
    /**print data of this tensor*/
    void PrintData();
    /**打印维度.*/
    void PrintShape();
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
    /**对标定维度求平均数.*/
    Tensor* Mean(std::vector<size_t>InputDims);
    /**对标定维度求方差.*/
    //Tensor* Var(std::vector<size_t>InputDims);
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
    /**高斯分布.*/
    void FillRandomValNormal();
    void FillRandomValNormal(float MeanV, float VarianceV);
    void FillRandomValNormal(float MeanV, float VarianceV,unsigned Seed);
    /**伯努利分布.*/
    void FillRandomValBernoulli(float P);
    void FillRandomValBernoulli(float P, unsigned Seed);
    /**均匀分布.*/
    void FillRandomValUniform();
    void FillRandomValUniform(float MinV, float MaxV);
    void FillRandomValUniform(float MinV, float MaxV, unsigned Seed);
    /**生成符号矩阵.*/
    Tensor* GenerateSignTensor();
    Tensor* ReLU();
    /**元素幂次.*/
    Tensor* Pow(float Exponent);
    /**改变张量的shape.*/
    Tensor* View(std::vector<size_t> OutputShape, int MinusOneIdx = -1);
    /**返回一个onthot张量.*/
    static Tensor* CreateOnehotTensor(std::vector<size_t> InputShape, std::vector<size_t>InputData, size_t TokenLength = 0, size_t DeviceNum = 0);
    /**对于元素的cos,sin.*/
    Tensor* Sin();
    Tensor* Cos();
    /**生成等差数列.*/
    static Tensor* ArithmeticSequence(std::vector<size_t> InputShape, float A1, float Arithmetic, size_t DeviceNum = 0);
};