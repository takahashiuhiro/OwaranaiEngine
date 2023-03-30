# README
## 本项目的配置  

Cuda Version:11.8  
Cuda Drive Version:522.25  
C++ 11  
理论上来说，只要装了nvcc/g++/cuda驱动版本合理就应该能跑起来  

##  项目运行方法  

cd build  
cmake ../  
make  
./OwaranaiEngineTest  
执行完这些不出意外就跑的起来  

## 项目目录  

OwaranaiEngine  项目根目录  
----AIMoudle  AI模块  
--------ComputationalGraph 计算图     
--------Layer 网络层  
--------Ops  算子  
--------Optimizer  优化器   
--------Helpers  杂项类  
--------LossFunction  用于写loss函数的地方  
----CommonMathMoudle  通用数学模块(包含几何与张量部分)  
--------Cuda  一些cuda的基础操作实现  
----TestFile  测试模块用的各种main文件  

## 最近的一些计划

1.跑起来一个简单的手写数字识别  
2.实现基础的ReduNet 


## 使用文档

#### 张量计算
位置: Tensor.h

###### 成员变量
size_t ShapeCount  Data的长度
float* DataCPU  指向数据在CPU位置的指针
float* DataGPU  指向数据在GPU位置的指针
std::string Device  标识张量目前所在的设备
size_t DeviceNum  张量的设备编号

###### 构造函数:
Tensor(){}
Tensor(std::vector<size_t>shape)
Tensor(std::vector<size_t>shape, std::string Device, size_t DeviceNum)  
使用张量的形态,设备和设备计数来初始化张量

###### 矩阵+标量(CPU+GPU):
Tensor* AddScalar(float Scalar)
参数输入一个标量的浮点数,返回一个和本身相加的张量

###### 矩阵+矩阵(CPU+GPU)
Tensor* Add(Tensor* Input)
参数输入一个和原本形态相同的矩阵,返回结果矩阵的指针

###### 矩阵*标量(CPU+GPU)
Tensor* MulScalar(float Scalar)
参数输入一个标量的浮点数,返回一个和本身相乘的张量

###### 矩阵广播乘(CPU+GPU)
Tensor* Matmul(Tensor* Input)
参数输入一个矩阵, 如果两个矩阵维度不相同,则缺少的部分由多的补齐,如果相同维度的形态不一,则该维度值形态值为1的那个矩阵复制后与另一个矩阵进行矩阵乘法

###### 矩阵转置(CPU+GPU)
Tensor* T()
返回原矩阵的转置

###### 矩阵元素乘(CPU+GPU)
Tensor* EleMul(Tensor* Input)
对输入矩阵进行元素乘后返回矩阵的指针

###### 按维度求平均值(CPU+GPU)
Tensor* AverageTensorDim(size_t InputDim)
指定一个维度将该维度所有的值取平均,其他维度不变,返回矩阵

###### 按维度求和(CPU+GPU)
Tensor* SumTensorDim(size_t InputDim)
指定一个维度将该维度所有的值求和,其他维度不变,返回矩阵

###### 取值(CPU+GPU)
float GetV(std::vector<size_t> FindIndex);
按照输入维度取值(GPU的情况下速度非常慢,仅供debug使用)

###### 赋值(CPU+GPU)
void SetV(std::vector<size_t> FindIndex, float Value)
按照输入维度赋值(GPU的情况下速度非常慢,仅供debug使用)

###### 填充矩阵(CPU+GPU)
void FillArray(float Scalar)
输入一个浮点数将矩阵中所有元素变为这个数

###### 打印矩阵(CPU+GPU)
void PrintData()
把矩阵按照从左到右从高维到低维的顺序打印

###### 单位矩阵(CPU+GPU)
Tensor* GetUnitTensor(std::vector<size_t>ReturnShape)
按照输入维度返回一个单位矩阵

###### 矩阵拼接(CPU+GPU)
Tensor* TensorSplice(Tensor* InputTensor, int SpliceDim)
输入一个矩阵并指定一个维度,两个矩阵从该维度进行拼接,返回一个新矩阵

###### 高斯消元(CPU+GPU)
void GaussianElimination()
对该张量按行进行高斯消元, 需要行数≤列数(虽然等于的时候会消出一个单位矩阵没什么用就是)

###### 设备更换
void ToGPU()
void ToCPU()
将该张量的数据从CPU移动到GPU或者从GPU移动到CPU
