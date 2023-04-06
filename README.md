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

### 张量计算
位置: Tensor.h  
如无标注则下述函数同时支持在张量在GPU或CPU上运算

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

###### 矩阵元素运算: 
Tensor* AddScalar(float Scalar)  
矩阵+标量, 参数输入一个标量的浮点数,返回一个和本身相加的张量  
Tensor* Add(Tensor* Input)  
矩阵+矩阵, 参数输入一个和原本形态相同的矩阵,返回结果矩阵的指针  
Tensor* MulScalar(float Scalar)  
矩阵*标量, 参数输入一个标量的浮点数,返回一个和本身相乘的张量  
Tensor* EleMul(Tensor* Input)  
矩阵元素乘, 对输入矩阵进行元素乘后返回矩阵的指针  

###### 矩阵广播乘
Tensor* Matmul(Tensor* Input)  
参数输入一个矩阵, 如果两个矩阵维度不相同,则缺少的部分由多的补齐,如果相同维度的形态不一,则该维度值形态值为1的那个矩阵复制后与另一个矩阵进行矩阵乘法  

###### 矩阵转置
Tensor* T()  
返回原矩阵的转置   

###### 按维度求平均值
Tensor* AverageTensorDim(size_t InputDim)  
指定一个维度将该维度所有的值取平均,其他维度不变,返回矩阵  

###### 按维度求和
Tensor* SumTensorDim(size_t InputDim)  
指定一个维度将该维度所有的值求和,其他维度不变,返回矩阵  

###### 张量内单值操作
float GetV(std::vector<size_t> FindIndex)  
取值, 按照输入维度取值(GPU的情况下速度非常慢,仅供debug使用)  
void SetV(std::vector<size_t> FindIndex, float Value)  
赋值, 按照输入维度赋值(GPU的情况下速度非常慢,仅供debug使用)  

###### 填充矩阵
void FillArray(float Scalar)  
输入一个浮点数将矩阵中所有元素变为这个数  

###### 打印矩阵
void PrintData()  
把矩阵按照从左到右从高维到低维的顺序打印  

###### 单位矩阵
Tensor* GetUnitTensor(std::vector<size_t>ReturnShape)  
按照输入维度返回一个单位矩阵  

###### 矩阵拼接
Tensor* TensorSplice(Tensor* InputTensor, int SpliceDim)  
输入一个矩阵并指定一个维度,两个矩阵从该维度进行拼接,返回一个新矩阵  

###### 高斯消元
void GaussianElimination()  
对该张量按行进行高斯消元, 需要行数≤列数(虽然等于的时候会消出一个单位矩阵没什么用就是)  

###### 设备更换
void ToGPU()  
void ToCPU()  
将该张量的数据从CPU移动到GPU或者从GPU移动到CPU  

###### 截取矩阵
Tensor* GetTensorBy2ShapeVector(std::vector<size_t>StartShape, std::vector<size_t>EndShape)
输入一个起点的索引和一个终点的索引，从中间抠出来对应的张量

###### 矩阵求逆
Tensor* Inverse()
返回该矩阵的逆，本质上是通过组合单位矩阵和原矩阵通过高斯消元后进行对应维度的截取得到的

### 计算图
位置: CGNode.h 

###### 成员变量
Tensor* NodeContent  该计算节点计算后指向的数据  
bool NeedGradient  该计算节点是否需要梯度  
std::vector<CGNode*>InputNode  上游计算节点  
CGNode* DerivativeNode  该计算节点指向的梯度计算节点  
BaseOps<CGNode,Tensor>* FunOps  该计算节点使用的算子  
std::string OpsType  该计算节点使用的算子类型  
std::map<std::string, bool>NodeType  标记该计算节点的节点类型,例如是否常量,是否冻结等  
bool BackwardBuildFlag  用于计算图反向过程中的记忆化标记  

###### 构造函数 
CGNode(){}  
声明一个计算节点
CGNode(bool NeedGradient)  
声明一个计算节点并标注是否需要梯度  
CGNode(Tensor* NodeContent, bool NeedGradient)  
声明一个计算节点并表明指向的张量以及是否需要梯度  
CGNode(std::string OpsType, bool NeedGradient)  
声明一个计算节点并表明算子类型以及是否需要梯度  
CGNode(std::string OpsType, bool NeedGradient, Hyperparameter OpsParams)  
声明一个计算节点并表明算子类型以及是否需要梯度,该计算节点需要输入的参数  
CGNode(std::vector<CGNode*>InputNode, std::string OpsType, bool NeedGradient)  
声明一个计算节点并表明构成该计算节点上游节点,算子类型,是否需要梯度  
CGNode(std::vector<CGNode*>InputNode, std::string OpsType, bool NeedGradient, Hyperparameter OpsParams)  
声明一个计算节点并表明构成该计算节点上游节点,算子类型,是否需要梯度,该计算节点需要输入的参数  

###### 设置算子
void SetOps(std::string OpsType)  
void SetOps(std::string OpsType, Hyperparameter OpsParams)  
输入算子类型,如果需要额外参数则需要输入第二个参数  

###### 计算前向
void Forward()  
通过该节点调用上游计算节点dfs式的得到结果,并保存到NodeContent中  

###### 反向过程
void BackwardBuild(bool IsOutput)  
通过该节点执行反向后会通过dfs计算上游计算节点,通过每个节点对应的算子执行反向构建, 该节点需要标记是否是输出节点以和上游节点做区分  
void Backward(Tensor* Loss)  
对该节点执行BackwardBuild并输入loss, 如果需要求出计算图中某个链的导数只需要对该链的起点的梯度节点求前向  

###### 数据清理
void ClearDataContent(std::vector<std::string>NodeTypeList, bool IsInclude)  
对该节点进行数据清理, 如果是True的话那就删除包含NodeTypeList内标签节点的content，False的话删除[不]包含的  
void ClearDataDFS(std::vector<std::string>NodeTypeList, bool IsInclude, std::map<CGNode*, bool>*FlagMap)
以该节点为起点对上游计算节点进行数据清理  
void ClearGradient(std::vector<CGNode*>InputNodeList)  
以该节点为起点对上游计算节点进行数据清理梯度数据  
void ClearComputeResult(std::vector<CGNode*>InputNodeList)  
以该节点为起点对上游计算节点进行数据清理常量和权重参数以外的量  

