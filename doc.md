# OwaranaiEngine's Doc

这是本项目的文档，api的说明请直接通过ctrl+f查阅 

## 1. File Structure

```
OwaranaiEngine  项目根目录  
    Code  代码  
        AutomaticDifferentiation  自动微分  
            Layers  神经网络层  
            Loss  误差层  
            Ops  算子  
            Optimizer  优化器  
            ComputationalGraph 计算图  
            ComputationalNode 计算图节点
            ForwardFunction 封装前向计算节点的函数，不含可训练参数
        CommonDataStructure  通用数据结构  
        CommonMathMoudle  通用数学  
            Cuda  一些cuda的基础操作实现  
            Tensor  张量逻辑   
            MathHelpers 基础数据结构实现的数学逻辑
    TestSample  测试模块用的各种main文件
```

## 2. Code Description

### 2.1. Tensor

本节主要说明张量部分的主要成员，接口与逻辑。

#### 2.1.1. Member Variables

```
std::shared_ptr<DevicePointerManager> DPMgr
用于数据管理的智能指针，在释放Tensor类的时候该指针负责释放申请的一切Device或Host资源
std::vector<size_t>shape
用于记录Tensor的shape
size_t ShapeCount
用于记录Tensor内的元素数量
```

#### 2.1.3. Member Functions

```
Tensor(){}
Tensor(std::vector<size_t>shape);
Tensor(std::vector<size_t>shape, size_t DeviceNum);
Tensor(std::vector<size_t>shape, size_t DeviceNum, std::vector<float> InputData);
Tensor(std::vector<size_t>shape, size_t DeviceNum, std::vector<float>* InputData);
函数说明：构造函数，输入参数均用于为成员变量赋值
```

```
Tensor* Copy();
拷贝一个与当前张量完全相同的张量
```

```
static Tensor* CreateTensorByLoadPath(std::ifstream& OpenedFile, size_t DeviceNum);
static Tensor* CreateTensorByLoadPath(std::ifstream& OpenedFile);
函数说明：通过打开的二进制文件创建张量
static Tensor* CreateTensorByLoadPath(std::string LoadPath, size_t DeviceNum);
static Tensor* CreateTensorByLoadPath(std::string LoadPath);
函数说明：通过二进制文件地址创建张量
```

```
size_t GetDeviceNum()
函数说明：使张量移动到指定设备编号
```

```
float* GetDevicePointer()
函数说明：得到张量的数据指针
```

```
void PrintData();
函数说明：按维度打印张量数据
```

```
void FillArray(float Scalar);
函数说明：使指定标量填充整个张量
参数说明：
    float Scalar：指定标量
```

```
size_t GetIndex(std::vector<size_t> FindIndex);
函数说明：通过输入的张量下标获取向量下标
参数说明：
    std::vector<size_t> FindIndex 输入的张量
```

```
Tensor* AddArray(Tensor* Input);
函数说明：与输入张量进行作为向量相加
参数说明：
    Tensor* Input 输入张量
```

```
Tensor* Add(Tensor* Input);
函数说明：与输入张量进行元素加
参数说明：
    Tensor* Input 输入张量
```

```
Tensor* EleMul(Tensor* Input);
函数说明：与输入张量进行元素乘
参数说明：
    Tensor* Input 输入张量
```

```
Tensor* AddScalar(float Scalar);
函数说明：张量内逐元素与输入标量相加
参数说明：
    float Scalar 输入标量
```

```
Tensor* MulScalar(float Scalar);
函数说明：张量内逐元素与输入标量相
参数说明：
    float Scalar 输入标量
```

```
Tensor* Matmul(Tensor* Input);
函数说明：与输入张量进行矩阵乘法，返回张量的最后两维与输入张量的最后两维有关，其他维度需要相等
参数说明：
    Tensor* Input 输入张量
```

```
Tensor* T();
函数说明：返回一个对本张量最后两维进行转置的张量
```

```
Tensor* Sum(std::vector<size_t>InputDims);
函数说明：对指定的多个输入维度求和，被求和的维度长度变为1
参数说明：
    std::vector<size_t>InputDims 多个指定维度
```

```
Tensor* Mean(std::vector<size_t>InputDims);
函数说明：对指定的多个输入维度求平均值，被求平均的维度长度变为1
参数说明：
    std::vector<size_t>InputDims 多个指定维度
```

```
Tensor* Var(std::vector<size_t>InputDims);
函数说明：对指定的多个输入维度求方差，被求方差的维度长度变为1（未实现）
参数说明：
    std::vector<size_t>InputDims 多个指定维度
```

```
Tensor* SumTensorDim(size_t InputDim);
函数说明：对指定的输入维度求和，被求和的维度长度变为1
参数说明：
    size_t InputDim 指定维度
```

```
Tensor* AverageTensorDim(size_t InputDim);
函数说明：对指定的输入维度求平均值，被求平均的维度长度变为1
参数说明：
    size_t InputDim 指定维度
```

```
void GaussianElimination();
函数说明：假设本张量为[X,I]的前提下做的高斯消元，其中X为待求解最矩阵，I为单位矩阵
求解后I的部分即为所求
```

```
Tensor* TensorSplice(Tensor* InputTensor, int SpliceDim);
函数说明：为本张量与输入张量通过某个维度进行拼接
参数说明：
    Tensor* InputTensor 待拼接张量
    int SpliceDim 通过该维度进行拼接
```

```
static Tensor* GetUnitTensor(std::vector<size_t>ReturnShape, size_t ReturnDeviceNum);
函数说明：为指定的shape与指定的设备生成一个单位矩阵
参数说明：
    std::vector<size_t>ReturnShape 单位矩阵的维度
    size_t ReturnDeviceNum 单位矩阵所在的
```

```
Tensor* GetTensorBy2ShapeVector(std::vector<size_t>StartShape, std::vector<size_t>EndShape);
函数说明：通过指定的两个张量下标从本张量中抠出一个新张量返回，要注意的使第二个参数的index要全部大于第一个参数
参数说明：
    std::vector<size_t>StartShape 指定下标1
    std::vector<size_t>EndShape 指定下标2
```

```
Tensor* Inverse();
函数说明：返回该矩阵的逆矩阵
```

```
Tensor* EleInverse();
函数说明：返回该矩阵的元素的逆组成的矩阵
```

```
Tensor* Maximum(size_t InputDim);
Tensor* Minimum(size_t InputDim);
函数说明：通过指定维度返回一个最大值或者最小值
参数说明：
    size_t InputDim 指定维度
```

```
Tensor* EleExp(float BaseNum);
函数说明：用输入值作为底数，张量内元素作为指数求幂
参数说明：
    float BaseNum 底数
```

```
Tensor* BroadCastTo(std::vector<size_t>BroadCastShape);
函数说明：把矩阵广播到指定的维度
参数说明：
    std::vector<size_t>BroadCastShape 待广播维度
```

```
bool CanBroadCastTo(std::vector<size_t>BroadCastShape);
函数说明：判断是否能把矩阵广播到指定的维度
参数说明：
    std::vector<size_t>BroadCastShape 待广播维度
```

```
Tensor* Softmax(size_t InputDim);
函数说明：对指定维度做softmax
参数说明：
    size_t InputDim 指定维度
```

```
void SaveToFile(std::ofstream& OpenedFile);
void SaveToFile(std::string FilePath);
void LoadFromFile(std::ifstream& OpenedFile);
void LoadFromFile(std::string FilePath);
函数说明：将张量数据和维度信息存储/提取到指定文件中
参数说明：
    std::ofstream& OpenedFile 指定二进制文件
    std::string FilePath 指定二进制文件地址
```

```
void FillRandomValNormal();
void FillRandomValNormal(unsigned Seed);
函数说明：用高斯分布填满张量
参数说明：
    unsigned Seed 随机种子
```

```
void FillRandomValBernoulli(float P);
void FillRandomValBernoulli(float P, unsigned Seed);
函数说明：用指定概率的Bernoulli分布填满张量
参数说明：
    float P 指定概率
    unsigned Seed 随机种子
```

```
Tensor* GenerateSignTensor();
函数说明：根据本张量生成一个符号张量，大于0的元素为1，小于0的元素为0
```

```
Tensor* ReLU();
函数说明：返回一个对张量的元素求ReLU的张量
```

```
Tensor* Pow(float Exponent);
函数说明：对张量的元素求幂
参数说明：
    float Exponent 指数
```

### 2.2. ComputationalGraph 
本节主要说明计算图部分的主要成员，接口与逻辑。 
#### 2.2.1. Member Variables
#### 2.2.3. Member Functions