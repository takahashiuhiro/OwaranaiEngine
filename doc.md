# OwaranaiEngine's Doc

这是本项目的文档，api的说明请直接通过ctrl+f查阅 

## 1. File Structure

```python
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

#### 2.2.1. Member Variables

```python
std::shared_ptr<DevicePointerManager> DPMgr
用于数据管理的智能指针，在释放Tensor类的时候该指针负责释放申请的一切Device或Host资源
std::vector<size_t>shape
用于记录Tensor的shape
size_t ShapeCount
用于记录Tensor内的元素数量
```

#### 2.2.3. Member Functions

```python
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
