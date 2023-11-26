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

本节主要说明Tensor类的主要成员，接口与逻辑。

#### 2.1.1. Member Variables

```
std::shared_ptr<DevicePointerManager> DPMgr
用于数据管理的智能指针，在释放Tensor类的时候该指针负责释放申请的一切Device或Host资源
```
```
std::vector<size_t>shape
用于记录Tensor的shape
```
```
size_t ShapeCount
用于记录Tensor内的元素数量
```

#### 2.1.2. Member Functions

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
函数说明：拷贝一个与当前张量完全相同的张量
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
本节主要说明ComputationalGraph类的主要成员，接口与逻辑。 
#### 2.2.1. Overview
ComputationalGraph类继承了BaseGraph类，拥有显式的图结构，因此在使用时也遵循普通有向图的用法，图中的node为需要特定id注册的ComputationalNode类，图中的edge在该结构中体现为node之间的邻接关系，并无权重体现。\
在进行前向使用时，需要对入度为0的node进行手动赋值，在需要输出的node执行dfs。在进行反向使用时，需要先对图进行反向建图，然后在反向图中的目标输出node执行前向dfs，因而该类中并无明确的反向计算流程。\
由于该类的结构设计，可以对图结构进行多次求导，直到在图中不存在对应阶导数的算子。
#### 2.2.2. Member Variables
```
using OpsMap = std::map<std::string, std::shared_ptr<BaseOps>>;
OpsMap Opss;
一个用来记录每个节点对应的算子的map
```
```
std::map<std::string, bool>ComputeFlag;
用于给dfs进行记忆化剪枝，且每个算子的反向建图只会执行一次，永不清空.
```
```
std::map<std::string, bool>BackwardFlag;
用于记录当前哪些节点需要被求导，本轮新增节点不求导.
```
```
std::map<std::string, size_t>CurrentBackwardFlag;
每个节点当前的求导轮数编号，用于在反向建图中区别不同阶的导数节点
size_t CurrentBackwardFlagIndex = 0;
计算图当前的求导轮数编号，用于在反向建图中区别不同阶的导数节点
```
```
bool CGMode  = true;
计算图模式，true为训练模式，false为推理模式.
```
#### 2.2.3. Member Functions
```
ComputationalNode* GetNode(std::string Nodeid);
函数说明：返回一个指定名字的ComputationalNode
参数说明：
    std::string Nodeid 指定的节点名字
```
```
void RegisterNode(std::string id);
void RegisterNode(std::string id,std::vector<size_t>ThisNodeShape);
void RegisterDefaultProperty(std::string Nodeid);
void RegisterDefaultProperty(std::string Nodeid, std::vector<size_t>ThisNodeShape);
函数说明：注册一个指定id的ComputationalNode，可以使用带有默认属性的节点
参数说明：
    std::string id 指定id
    std::vector<size_t>ThisNodeShape 指定节点对应的张量维度
```

```
void RegisterVariableNode(std::string Nodeid);
void RegisterVariableNode(std::string Nodeid, std::vector<size_t>ThisNodeShape);
void RegisterWeightNode(std::string Nodeid);
void RegisterWeightNode(std::string Nodeid, std::vector<size_t>ThisNodeShape);
void RegisterConstNode(std::string Nodeid);
void RegisterConstNode(std::string Nodeid, std::vector<size_t>ThisNodeShape);
函数说明：RegisterVariableNode在RegisterNode的基础上注册默认带有梯度的变量节点，RegisterWeightNode在变量的基础上增加无法被清理梯度的函数清理的属性，RegisterConstNode不可导，也不可被清理
参数说明：
    std::string id 指定id
    std::vector<size_t>ThisNodeShape 指定节点对应的张量维度
```
```
void RegisterOps(std::string OutputNodeid, std::vector<std::string> InputNodeid, size_t OpsTypeid, Dict OpsParams);
函数说明：注册一个算子，算子需要和节点同名，并为输入节点建立邻接关系。
参数说明：
    std::string OutputNodeid 算子所在的节点id
    std::vector<std::string> InputNodeid 输入节点的id
    size_t OpsTypeid 算子类型
    Dict OpsParams 算子初始化属性
```

```
void RegisterOpsAddEdge(std::string OutputNodeid, std::string InputNodeid);
函数说明：为指定算子增加一个输入节点，作用于无法一次性注册完的算子
参数说明：
    std::string OutputNodeid 算子所在的节点id
    std::string InputNodeid 新增输入节点
```
```
void RegisterOpsCompleted(std::string OutputNodeid, std::vector<std::string> InputNodeid, size_t OpsTypeid, Dict OpsParams);
函数说明：一次性注册完整个算子，不再进行边的增删，并且附加默认属性
参数说明：
    std::string OutputNodeid 算子所在的节点id
    std::vector<std::string> InputNodeid 输入节点的id
    size_t OpsTypeid 算子类型
    Dict OpsParams 算子初始化属性
```
```
void SetOpsInputNodeDefaultParams(std::string OutputNodeid);
函数说明：给算子内的输入节点赋予默认参数
参数说明：
    std::string OutputNodeid 算子所在的节点id
```
```
void BackwardGraphBuild();
函数说明：建立反向图
```
```
void RegisterDNode(std::string id);
函数说明：为指定节点注册其梯度节点
参数说明：
    std::string id 指定节点id
```
```
std::string GetDNodeid(std::string id);
函数说明：为指定节点寻找其梯度节点的id
参数说明：
    std::string id 指定节点id
```
```
std::string GetCopyNode(std::string id);
函数说明：复制一个指定id的节点
参数说明：
    std::string id 指定节点id
```
```
std::string GetDPartNodeid(std::string Startid, std::string Endid);
函数说明：在算子里会出现a->c且a->b的情况,在这种情况下如果c对a求导，会在a_d和c_d之间的反向过程里建立一个新节点，用来表达c_d给a_d的贡献的中间节点，已弃用。
```
```
std::string GetNodeidByOps(size_t OpsName, std::vector<std::string>InputNodeNameArray);
函数说明：通过算子类型，输入节点，以及时间种子生成一个新的节点id
参数说明：
    size_t OpsName 算子类型
    std::vector<std::string>InputNodeNameArray 输入节点
```
```
void ForwardDfs(std::string StartNodeid);
函数说明：对指定id的节点通过dfs进行前向计算
参数说明：
    std::string StartNodeid 指定节点id
```
```
void NodeOpsForward(std::string DfsStartNodeid);
函数说明：对指定id的算子进行前向计算，一般不单独调用
参数说明：
    std::string DfsStartNodeid 指定节点id
```

```
std::shared_ptr<BaseOps> GetCGOps(std::string OpsNodeid);
函数说明：返回一个指定id的算子
参数说明：
    std::string OpsNodeid 指定的算子id
```
```
void ClearAllData();
函数说明：清除计算图中所有数据
```
```
void PrintGraphAdjacencyList(size_t Mode);
函数说明：打印计算图的邻接表
参数说明：
    size_t Mode 值为1的时候输出输入节点，2的时候输出输出节点，3的时候两者都输出
```
```
void BackwardMultiBuildGraph(size_t Times);
函数说明：对计算图求多阶导数
参数说明：
    size_t Times 求导次数
```
```
void ClearDataPropertyExclude(std::vector<std::string>CheckPropertyList);
函数说明：指定词条任意为false的节点将被清理数据
参数说明：
    std::vector<std::string>CheckPropertyList 指定词条
```
```
void ClearWeightConstExclude();
函数说明：清除Weight和Const以外的节点.
```
```
std::vector<std::string> GetNodesByProperty(std::vector<std::string>IncludeList, std::vector<std::string>ExcludeList);
函数说明：按需求查询图内节点
参数说明：
    std::vector<std::string>IncludeList 需要包含的属性
    std::vector<std::string>ExcludeList 不能包含的属性
```
```
void ComputeWeightNodesDForward();
函数说明：对所有需要求导的节点求导
```
```
void SetTrainMode();
void SetEvalMode();
函数说明：设置计算图模式，用于dropout之类的逻辑
```

### 2.3. BaseOps
本节主要介绍基算子，派生算子除计算逻辑外无不同
#### 2.3.1. Overview
在本框架中，算子的作用是链接不同的计算节点组成计算图，在对某个节点求前向计算后，会调用计算图内所有被dfs搜索到的算子进行顺序计算得到结果。
#### 2.3.2. Member Variables
```
size_t OpsTypeName;
算子类型
```
```
Dict Params;
算子的参数，例如转置，系数等信息
```
```
ComputationalGraph* CG;
算子对应的计算图
```
```
std::string Nodeid;
算子在计算图中的id
```
#### 2.3.3. Common Member Functions
```
void ForwardProcess();
函数说明：计算图调用该函数进行算子的前向计算，该函数会调用虚函数Forward()进行具体的各个派生类的前向计算，然后统一调用后处理计算，例如dropout
```
```
void CommonInit(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG);
函数说明：算子通过类型，参数，计算图被初始化
参数说明：
    size_t OpsTypeName 算子类型
    Dict Params 算子参数
    ComputationalGraph* ParentCG 算子所在的计算图
```
```
std::vector<std::string> &GetInputNodeList();
std::vector<std::string> &GetInputNodeList(std::string InputNodeid);
函数说明：获取节点的输入列表
参数说明：
    std::string InputNodeid 指定节点id，没有id输入则默认id为本算子id
```
```
void CGForwardProcess();
函数说明：计算图标记后处理,只有在前向赋值计算以后才允许调用
```
```
void CGForwardProcessDropout();
函数说明：在前向计算有结果后进行dropout逻辑判断与计算
```
#### 2.3.4. Virtual Member Functions
```
virtual void Forward() = 0;
函数说明：派生类会重载这个函数进行前向计算
```
```
virtual void Backward();
函数说明：派生类会重载这个函数进行反向计算
```
```
virtual void AfterSettingShapeComputing() = 0;
函数说明：派生类会重载这个函数进行节点的张量所需形状进行计算
```
### 2.4. BaseLayer
本节内容为封装计算图中的带有可训练参数的节点声明，用于加载模型权重，保存模型权重，复用计算图搭建逻辑
#### 2.4.1 Overview
在本框架中，一个模型的所有Layer共用同一个计算图，主Layer会自动声明一个计算图，如果有Layer逻辑复用，则会独立与计算图外建立一套Layer的树关系，在保存权重的时候通过截断节点名称的树链对权重节点进行保存。读取的时候会按照树在模型的层级增加节点名称进行读取。
#### 2.4.2 Member Variables
```
std::string PreName = "";
当前子树的前置链名
```
```
std::string LayerName = "";
该层的相对名字
```
```
std::map<std::string, std::shared_ptr<BaseLayer>>SubLayers;
本层子树的直接孩子
```
```
std::vector<std::string>WeightNodeArray;
本层所属的直接注册的权重矩阵，孩子的权重矩阵不在其中
```
```
std::shared_ptr<ComputationalGraph> CG = nullptr;
模型共用的计算图
```
```
BaseLayer* ParentLayer = nullptr;
父节点，root为nullptr
```
```
size_t DeviceNum = 0;
设备数(同tensor的设备数)
```
```
std::vector<std::string> InputNodes;
在本层注册的输入节点
```
#### 2.4.3. Common Member Functions
```
void CommonInit(BaseLayer* InputParentLayer, std::string InputLayerName, size_t ThisDeviceNum);
函数说明：公共init，例如在构造函数的时候声明计算图等.调用这个函数前要先把该分给子层的分了，防止重复声明计算图，寄了
参数说明：
    BaseLayer* InputParentLayer 该层的父级指针
    std::string InputLayerName 该层的相对名字
    size_t ThisDeviceNum 该层的设备号，用于分配张量设备
```
```
void RegisterLayer(std::shared_ptr<BaseLayer>InputLayer);
函数说明：注册子Layer
参数说明：
    std::shared_ptr<BaseLayer>InputLayer 子Layer的指针
```
```
void RegisterWeightNode(std::string InputNodeid,std::vector<size_t>InputTensorShape);
void RegisterInputNode(std::string InputNodeid,std::vector<size_t>InputTensorShape);
void RegisterConstNode(std::string InputNodeid,std::vector<size_t>InputTensorShape);
函数说明：转发计算图中的同名成员函数，注册权重，输入，常量节点
```
```
std::string GetLayerNodeName(std::string InputNodeName);
函数说明：通过层内相对节点名字获取在计算图中的绝对名字
参数说明：
    std::string InputNodeName 层内相对名字
```
```
void SaveToFile(std::string SavePath);
void LoadFromFile(std::string LoadPath);
函数说明：保存权重和加载权重
参数说明：
    std::string SavePath 保存权重的二进制文件地址
    std::string LoadPath 加载权重的二进制文件地址
```
```
std::vector<std::string> GetAllSubLayersNodeDfs();
std::vector<std::string> GetAllSubLayersNodeDfs(bool AutoFlag);
函数说明：dfs搜出子树所有需要保存的Layer
```
#### 2.4.4. Virtual Member Functions
```
virtual std::vector<std::string> Forward(std::vector<std::string>InputNodeArray){{}};
函数说明：负责dfs的前向构建该子树下的计算图
参数说明：
    std::vector<std::string>InputNodeArray 本层的输入节点
```

### 2.5. OEAutoDiff
本节主要介绍了OEAutoDiff类中函数的参数和作用
#### 2.5.1 Overview
由于图操作非常繁琐且容易出错，因此在该类中封装了静态函数用来代替图操作做一些不会声明权重矩阵的前向计算图，封装操作的第一个参数必须为ComputationalGraph*CG 计算图，因此后续不再赘述
#### 2.5.2 Member Functions
```
static std::string Add(ComputationalGraph*CG,std::map<std::string, float> InputWeight);
函数说明：矩阵加法
参数说明：
    std::map<std::string, float> InputWeight 需要相加的节点名以及对应的系数
```
```
static std::string Pow(ComputationalGraph*CG,std::string InputNode,float Exponent);
函数说明：矩阵元素求幂
参数说明：
    std::string InputNode 输入节点
    float Exponent 幂次
```
```
static std::string BroadCastTo(ComputationalGraph*CG,std::string InputNode,std::vector<size_t>InputDims);
函数说明：矩阵广播
参数说明：
    std::string InputNode 输入节点
    std::vector<size_t>InputDims 目标广播维度
```
```
static std::string EleMul(ComputationalGraph*CG,std::string FirstNode, std::string SecondNode,float FirstAddWeight = 1., float SecondAddWeight = 1.);
函数说明：矩阵元素乘
参数说明：
    std::string FirstNode 第一项节点名
    std::string SecondNode 第二项节点名
    float FirstAddWeight = 1 第一项节点系数默认为1
    float SecondAddWeight = 1 第二项节点系数默认为1
```
```
static std::string MatMul(ComputationalGraph*CG, std::string FirstNode, std::string SecondNode, bool FirstTFlag = false, bool SecondTFlag = false, float FirstAddWeight = 1., float SecondAddWeight = 1.);
函数说明：矩阵乘法
参数说明：
    std::string FirstNode 第一项节点名
    std::string SecondNode 第二项节点名
    bool FirstTFlag = false 第一项是否转置
    bool SecondTFlag = false 第二项是否转置
    float FirstAddWeight = 1 第一项节点系数默认为1
    float SecondAddWeight = 1 第二项节点系数默认为1
```
```
static std::string Sum(ComputationalGraph*CG,std::string InputNode, std::vector<size_t>InputDims);
函数说明：指定维度求和
参数说明：
    std::string InputNode 输入节点
    std::vector<size_t>InputDims 指定维度
```
```
static std::string Mean(ComputationalGraph*CG,std::string InputNode, std::vector<size_t>InputDims);
函数说明：指定维度求平均
参数说明：
    std::string InputNode 输入节点
    std::vector<size_t>InputDims 指定维度
```
```
static std::string Var(ComputationalGraph*CG,std::string InputNode, std::vector<size_t>InputDims,bool Unbiased = true);
函数说明：指定维度求方差
参数说明：
    std::string InputNode 输入节点
    std::vector<size_t>InputDims 指定维度
    bool Unbiased = true 是否是样本方差
```
```
static std::string ReLU(ComputationalGraph*CG,std::string InputNode);
函数说明：Relu激活函数
参数说明：
    std::string InputNode 输入节点
```
```
static std::string Softmax(ComputationalGraph*CG,std::string InputNode, size_t InputDim);
函数说明：Softmax
参数说明：
    std::string InputNode 输入节点
    size_t InputDim 指定维度
```
```
static std::string Dropout(ComputationalGraph*CG,std::string InputNode,float P ,bool InPlace = false);
函数说明：Dropout
参数说明：
    std::string InputNode 输入节点
    float P 神经元停止工作概率P
    bool InPlace = false 未实现
```
```
static std::string EleExp(ComputationalGraph*CG,std::string InputNode,float BaseNum);
函数说明：输入一个底数给出以矩阵元素为指数的张量
参数说明：
    std::string InputNode 输入节点
    float BaseNum 输入底数
```
```
static std::string Tanh(ComputationalGraph*CG,std::string InputNode);
函数说明：双曲正切激活函数
参数说明：
    std::string InputNode 输入节点
```
```
static std::string GELU(ComputationalGraph*CG,std::string InputNode);
函数说明：GELU激活函数
参数说明：
    std::string InputNode 输入节点
```