#pragma once
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <set>
#include "Tensor.h"
#include "OpsType.h"

namespace OwaranaiEngine
{

class DynamicTensor;

struct DynamicOps
{

    ~DynamicOps();

    /**算子成员变量.*/
    he Params;//算子参数
    size_t DynamicOpsType;//算子类型，叶子节点为Base
    DynamicTensor* LeafNode = nullptr;//指向算子的结算张量，如果为nullptr代表张量被删除
    std::vector<std::shared_ptr<DynamicOps>>InputOpsList;//输入节点，需要后继保存输入的资源
    std::set<DynamicOps*>OutputOpsSet;//输出节点，不需要保存output的资源
    std::shared_ptr<DynamicOps>GradOps = nullptr;
    std::shared_ptr<Tensor> TensorPointer = nullptr;//存的张量内容
    bool RequiresGrad = false;//是否需要求导
    bool IsEval = false;//如果有一个所有后续算子停止求导, 如果有网络层要开就从权重矩阵的这里开是否求导
};

class DynamicTensor
{
public:

    /**动态张量的成员变量.*/
    std::shared_ptr<DynamicOps>Ops = nullptr;//每个动态张量的算子，如果张量被删掉但是需要计算图，这个算子可以交出去，交出去的时候需要删掉算子中的leafNode变量为nullptr
    std::map<size_t, void(*)(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& ,std::shared_ptr<DynamicOps>)>BackwardOps;//反向函数map

    /**内存管理.*/
    DynamicTensor();//初始化动态张量
    DynamicTensor(std::shared_ptr<Tensor> InputTensorPointer, bool InputRequiresGrad = 0);
    DynamicTensor(std::shared_ptr<DynamicOps>InputOps);
    DynamicTensor(std::vector<size_t>InputShape, bool InputRequiresGrad = 0, size_t DeviceNum = 0);
    DynamicTensor(std::vector<size_t>InputShape, std::vector<float>InputData,bool InputRequiresGrad = 0, size_t DeviceNum = 0);
    static DynamicTensor CreateVector(std::vector<float>InputData, bool InputRequiresGrad = 0, size_t DeviceNum = 0);
    void OpsSetInMap();

    ~DynamicTensor();//析构函数释放内存

    /**公共函数.*/
    DynamicTensor Grad();
    /**复制一个空的只有张量的值. */
    DynamicTensor Copy();
    std::vector<size_t>& Shape();
    std::vector<int> ShapeInt();
    /**返回整个张量的元素个数 */
    int ShapeCount();

    void SetRequiresGrad(bool InputRequiresGrad);//使向量可导或者不可导

    /**Tensor内函数组装.*/
    //打印数据
    void PrintData();
    //填充标量
    void Fill(float InputValue);
    //填充伯努利分布
    void FillRandBernoulli(float P, int Seed = -1);
    //填充均匀分布
    void FillRandValUniform(float MinV = 0, float MaxV = 1, int Seed = -1);
    //填充高斯分布
    void FillRandomValNormal(float MeanV = 0, float VarianceV = 1,int Seed = -1);
    //得到设备号
    size_t GetDeviceNum();
    //创建一个onehot张量
    static DynamicTensor CreateOnehotTensor(std::vector<int> InputShape, std::vector<int>InputData, int TokenLength = 0,bool RequiresGrad = false, size_t DeviceNum = 0);
    //返回一个tensor的参数量
    int Numel();
    //创建一个新的等差数列张量
    static DynamicTensor Arange(float Start, float End, float Step = 1, bool RequiresGrad = false, size_t DeviceNum = 0);
    static DynamicTensor CreateUnitTensor(std::vector<int>ReturnShape, bool RequiresGrad = false, size_t DeviceNum = 0);


    /**计算图逻辑.*/
    static DynamicTensor SetComputationalHistory(Tensor* ResTensor, std::vector<DynamicTensor>InputList, he InputPrams,size_t InputOpsType, bool RequiresGrad);
    void Backward(DynamicTensor Loss = DynamicTensor(), bool ClearGrad = true);//从这里开始反向传播
    void BackwardDFS(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>&BackwardOpsMap, std::map<DynamicOps*, std::set<DynamicOps*>>& OutputSetSize, DynamicTensor Loss, std::shared_ptr<DynamicOps>CurOps);
    void BackwardClearDFS(std::shared_ptr<DynamicOps>CurOps);
    bool CheckPartialGradReady(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::map<DynamicOps*, std::set<DynamicOps*>>& OutputSetSize, std::shared_ptr<DynamicOps>CurOps);
    void GenEmptyGradDynamicTensor(DynamicTensor Loss);
    void GetAllOutputSizeBeforeBackward(std::map<DynamicOps*, std::set<DynamicOps*>>& OutputSetSize,std::shared_ptr<DynamicOps>CurOps);

    /**运算符重载逻辑.*/
    DynamicTensor operator+(DynamicTensor Other);
    DynamicTensor operator+(float Other);
    DynamicTensor operator%(DynamicTensor Other);//矩阵乘法
    DynamicTensor operator*(DynamicTensor Other);
    DynamicTensor operator*(float Other);
    DynamicTensor operator-(DynamicTensor Other);
    DynamicTensor operator-(float Other);

    /**重复逻辑抽出.*/
    DynamicTensor ViewAndBC(DynamicTensor ThisDT, DynamicTensor Other, DynamicTensor(*InputFun)(std::vector<DynamicTensor>, he, bool), bool IsMatmul);

    /**算子.*/
    static DynamicTensor DynamicStdOps_Forward_Add(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad = false);
    static void DynamicStdOps_Backward_Add(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>&BackwardOpsMap,std::shared_ptr<DynamicOps>CurOps);

    static DynamicTensor DynamicStdOps_Forward_Matmul(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad = false);
    static void DynamicStdOps_Backward_Matmul(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>&BackwardOpsMap,std::shared_ptr<DynamicOps>CurOps);

    static DynamicTensor DynamicStdOps_Forward_BroadCastTo(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad = false);
    static void DynamicStdOps_Backward_BroadCastTo(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps);

    static DynamicTensor DynamicStdOps_Forward_Sum(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad = false);
    static void DynamicStdOps_Backward_Sum(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps);

    static DynamicTensor DynamicStdOps_Forward_View(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad = false);
    static void DynamicStdOps_Backward_View(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps);

    static DynamicTensor DynamicStdOps_Forward_Elemul(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad = false);
    static void DynamicStdOps_Backward_Elemul(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps);

    static DynamicTensor DynamicStdOps_Forward_Softmax(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad = false);
    static void DynamicStdOps_Backward_Softmax(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps);

    static DynamicTensor DynamicStdOps_Forward_Pow(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad = false);
    static void DynamicStdOps_Backward_Pow(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps);

    static DynamicTensor DynamicStdOps_Forward_Eleexp(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad = false);
    static void DynamicStdOps_Backward_Eleexp(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps);

    static DynamicTensor DynamicStdOps_Forward_Transpose(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad = false);
    static void DynamicStdOps_Backward_Transpose(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps);

    static DynamicTensor DynamicStdOps_Forward_EleLog(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad = false);
    static void DynamicStdOps_Backward_EleLog(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps);

    static DynamicTensor DynamicStdOps_Forward_SubSend(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad = false);
    static void DynamicStdOps_Backward_SubSend(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps);

    /**可导函数.*/

    // 求和
    DynamicTensor Sum(std::vector<int>Dims = {}, bool KeepDim = false);
    DynamicTensor View(std::vector<int>Dims);
    DynamicTensor Softmax(int InputDim);
    DynamicTensor Pow(float EleExponent);
    static DynamicTensor Dropout(DynamicTensor Input, float P, bool InPlace = false);
    std::vector<DynamicTensor> Split(int SplitSize, int Dim = 0);
    std::vector<DynamicTensor> Split(std::vector<int> SplitSections, int Dim = 0);
    DynamicTensor Eleexp(float EleBaseNum);
    DynamicTensor Tanh();
    static DynamicTensor Cat(std::vector<DynamicTensor>InputTensors, int Dim = 0);
    DynamicTensor GELU();
    DynamicTensor ReLU();
    DynamicTensor Mean(std::vector<int>InputDims, bool KeepDim = false);
    // 方差
    DynamicTensor Var(std::vector<int>InputDims, bool KeepDim = false, float Correction = 1);
    // 除下三角外的部分都为0
    DynamicTensor Tril(int Diagonal = 0);
    DynamicTensor Transpose(int Dim0, int Dim1, int DebugFlag = false);
    DynamicTensor MaskedFill(DynamicTensor Mask, float Value);
    DynamicTensor EleLog();
    static DynamicTensor CrossEntropy(DynamicTensor Input, DynamicTensor Target, std::string Reduction = "Mean", DynamicTensor Weight = DynamicTensor(), float LabelSmoothing =0);
    DynamicTensor Sigmoid();
    //高斯分布的积分，默认的是期望是0标准差是1的标准高斯分布，用泰勒展开3项
    DynamicTensor GaussianCdf(float InputMean = 0, float InputStd =1, int Terms = 3);
    /**
     * 给出一个绝对值的tensor
     */
    DynamicTensor Abs();

    /**----------------------------不可导函数，包含能导但暂时用不到就没写的.---------------------------- */

    /**方阵分解为下三角矩阵 */
    DynamicTensor Cholesky(); //能导
    /**
     * 从标准高斯分布采样
     * @param Dim 高斯的维度
     * @param InputVec 输出的形状，不包含高斯本身
     * @param Seed 随机种子
     * @param DeviceNum 设备号
     */
    static DynamicTensor SampleFromStdGaussian(int Dim, std::vector<int> InputVec, int Seed = -1,int DeviceNum = 0);
    /**
     * 从非标准高斯分布采样
     * @param Dim 高斯的维度
     * @param InputVec 输出的形状，不包含高斯本身
     * @param Mean 均值
     * @param Var 协方差矩阵
     * @param Seed 随机种子
     * @param DeviceNum 设备号
     */
    static DynamicTensor SampleFromOtherGaussian(int Dim, std::vector<int> InputVec, DynamicTensor Mean, DynamicTensor Var, DynamicTensor VarL = DynamicTensor(), int Seed = -1,int DeviceNum = 0);

    /**矩阵求逆 理论上是应该可导的,但是现在没有实现成可导的. */
    DynamicTensor Inverse();

    /**求Det系列应该也是可导的 */
    /**对称矩阵求行列式，需要先进行LU分解 */
    DynamicTensor Det_Symmetric(DynamicTensor InputL);

    /**对采样计算概率密度. */
    static DynamicTensor ProbabilityDensity_Gaussian(DynamicTensor InputSample, DynamicTensor InputMean, DynamicTensor InputVarInv, DynamicTensor InputVarDet);
    /**上面的取log. */
    static DynamicTensor ProbabilityDensity_Log_Gaussian(DynamicTensor InputSample, DynamicTensor InputMean, DynamicTensor InputVarInv, DynamicTensor InputVarDet);

    /**求max, min这种不可导的. */
    DynamicTensor Max(std::vector<int>Dims = {}, bool KeepDim = false);
};

DynamicOps::~DynamicOps()
{
	for (size_t a = 0; a < InputOpsList.size(); a++)
	{
		if (InputOpsList[a]->OutputOpsSet.find(this) != InputOpsList[a]->OutputOpsSet.end())
		{
			InputOpsList[a]->OutputOpsSet.erase(this);
		}
	}
}

DynamicTensor::DynamicTensor()
{
	OpsSetInMap();
};
DynamicTensor::DynamicTensor(std::shared_ptr<Tensor> InputTensorPointer, bool InputRequiresGrad)
{
	Ops = std::make_shared<DynamicOps>();
	Ops->TensorPointer = InputTensorPointer;
	Ops->RequiresGrad = InputRequiresGrad;
	Ops->LeafNode = this;
	OpsSetInMap();
}
DynamicTensor::DynamicTensor(std::shared_ptr<DynamicOps>InputOps)
{
	Ops = InputOps;
	Ops->LeafNode = this;
	OpsSetInMap();
}

DynamicTensor::DynamicTensor(std::vector<size_t>InputShape, bool InputRequiresGrad, size_t DeviceNum)
{
	Ops = std::make_shared<DynamicOps>();
	Ops->TensorPointer = std::make_shared<Tensor>(InputShape, DeviceNum);
	Ops->RequiresGrad = InputRequiresGrad;
	Ops->LeafNode = this;
	OpsSetInMap();
}

DynamicTensor::DynamicTensor(std::vector<size_t>InputShape, std::vector<float>InputData, bool InputRequiresGrad, size_t DeviceNum)
{
	Ops = std::make_shared<DynamicOps>();
	Ops->TensorPointer = std::make_shared<Tensor>(InputShape, DeviceNum, InputData);
	Ops->RequiresGrad = InputRequiresGrad;
	Ops->LeafNode = this;
	OpsSetInMap();
}

DynamicTensor DynamicTensor::CreateVector(std::vector<float>InputData, bool InputRequiresGrad, size_t DeviceNum)
{
	std::vector<size_t>InputShape = { 1,InputData.size() };
	return DynamicTensor(InputShape, InputData, InputRequiresGrad, DeviceNum);
}

DynamicTensor::~DynamicTensor()
{
	if(Ops != nullptr)
	{
		if(Ops->LeafNode == this)Ops->LeafNode = nullptr;
	}
}

DynamicTensor DynamicTensor::Grad()
{
	Log::Assert(Ops->GradOps != nullptr, "This DynamicTensor Has No Grad");
	return DynamicTensor(Ops->GradOps);
}

DynamicTensor DynamicTensor::Copy()
{
	DynamicTensor Res(std::shared_ptr<Tensor>(Ops->TensorPointer->Copy()), 0);
	return Res;
}

std::vector<size_t>& DynamicTensor::Shape()
{
	Log::Assert(Ops != nullptr, "this ops of dynamictensor is nullptr, so you can't read its shape..");
	return Ops->TensorPointer->shape;
}

std::vector<int> DynamicTensor::ShapeInt()
{
	Log::Assert(Ops != nullptr, "this ops of dynamictensor is nullptr, so you can't read its shapeint..");
	std::vector<int>Res;
	for(auto&it:Ops->TensorPointer->shape)Res.push_back(it);
	return Res;
}

int DynamicTensor::ShapeCount()
{
	return Ops->TensorPointer->ShapeCount;
}

void DynamicTensor::SetRequiresGrad(bool InputRequiresGrad)
{
	Ops->RequiresGrad = InputRequiresGrad;
}

void DynamicTensor::PrintData()
{
	Ops->TensorPointer->PrintData();
}

void DynamicTensor::Fill(float InputValue)
{
	Ops->TensorPointer->FillArray(InputValue);
}

void DynamicTensor::FillRandBernoulli(float P, int Seed)
{
	if (Seed == -1)Seed = std::chrono::system_clock::now().time_since_epoch().count();
	Ops->TensorPointer->FillRandomValBernoulli(P, Seed);
}

void DynamicTensor::FillRandValUniform(float MinV, float MaxV, int Seed)
{
	if (Seed == -1)Seed = std::chrono::system_clock::now().time_since_epoch().count();
	Ops->TensorPointer->FillRandomValUniform(MinV, MaxV, Seed);
}

void DynamicTensor::FillRandomValNormal(float MeanV, float VarianceV,int Seed)
{
	if (Seed == -1)Seed = std::chrono::system_clock::now().time_since_epoch().count();
	Ops->TensorPointer->FillRandomValNormal(MeanV, VarianceV, Seed);
}

size_t DynamicTensor::GetDeviceNum()
{
	return Ops->TensorPointer->GetDeviceNum();
}

DynamicTensor DynamicTensor::CreateOnehotTensor(std::vector<int> InputShape, std::vector<int>InputData, int TokenLength,bool RequiresGrad, size_t DeviceNum)
{
	std::vector<size_t>ConvInputShape;
	for(auto&it:InputShape)ConvInputShape.push_back(it);
	std::vector<size_t>ConvInputData;
	for(auto&it:InputData)ConvInputData.push_back(it);
	return DynamicTensor(std::shared_ptr<Tensor>(Tensor::CreateOnehotTensor(ConvInputShape, ConvInputData, TokenLength, DeviceNum)), 0);
}

int DynamicTensor::Numel()
{
	return Ops->TensorPointer->ShapeCount;
}

DynamicTensor DynamicTensor::Arange(float Start, float End, float Step, bool RequiresGrad, size_t DeviceNum)
{
	auto TensorContent = Tensor::ArithmeticSequence({size_t((End-1e-6 - Start)/Step)+1}, Start, Step, DeviceNum);
	DynamicTensor Res(std::shared_ptr<Tensor>(TensorContent), RequiresGrad);
	return Res;
}

DynamicTensor DynamicTensor::CreateUnitTensor(std::vector<int>ReturnShape, bool RequiresGrad, size_t DeviceNum)
{
    Tensor* TensorContent;
    std::vector<size_t>ReturnShapeInt;
    for(auto&it:ReturnShape)ReturnShapeInt.push_back(it);
    TensorContent = Tensor::GetUnitTensor(ReturnShapeInt, DeviceNum);
    return DynamicTensor(std::shared_ptr<Tensor>(TensorContent), RequiresGrad);
}

DynamicTensor DynamicTensor::SetComputationalHistory(Tensor* ResTensor, std::vector<DynamicTensor>InputList, he InputPrams, size_t InputOpsType, bool RequiresGrad)
{
	bool MaxRequiresGrad = 0,MaxIsEval = 0;
	for (size_t a = 0; a < InputList.size(); a++)MaxRequiresGrad |= InputList[a].Ops->RequiresGrad;
	for (size_t a = 0; a < InputList.size(); a++)MaxIsEval |= InputList[a].Ops->IsEval;
	DynamicTensor Res(std::shared_ptr<Tensor>(ResTensor), MaxRequiresGrad&RequiresGrad&(!MaxIsEval));
	Res.Ops->IsEval = MaxIsEval;
	if ((!RequiresGrad)|| MaxIsEval)return Res;
	Res.Ops->DynamicOpsType = InputOpsType;
	Res.Ops->Params = InputPrams;
	for (size_t a = 0; a < InputList.size(); a++)
	{
		Res.Ops->InputOpsList.push_back(InputList[a].Ops);
		InputList[a].Ops->OutputOpsSet.insert(Res.Ops.get());
	}
	return Res;
}

void DynamicTensor::Backward(DynamicTensor Loss,bool ClearGrad)
{
	Log::Assert(Ops->OutputOpsSet.size() == 0, "DynamicTensor Backward Must Be Output Data");
	std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>BackwardOpsMap;
	std::map<DynamicOps*, std::set<DynamicOps*>>OutputSetSize;
	GetAllOutputSizeBeforeBackward(OutputSetSize, Ops);
	BackwardDFS(BackwardOpsMap, OutputSetSize,Loss, Ops);
	if (ClearGrad)BackwardClearDFS(Ops);
}

void DynamicTensor::BackwardClearDFS(std::shared_ptr<DynamicOps>CurOps)
{
	if (CurOps->InputOpsList.size())CurOps->GradOps = nullptr;
	else
	{
		if(CurOps->RequiresGrad)
		{
			CurOps->GradOps->InputOpsList = {};
			CurOps->GradOps->OutputOpsSet = {};
		}
	}
	for (size_t a = 0; a < CurOps->InputOpsList.size(); a++)BackwardClearDFS(CurOps->InputOpsList[a]);
}

void DynamicTensor::BackwardDFS(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::map<DynamicOps*, std::set<DynamicOps*>>& OutputSetSize, DynamicTensor Loss, std::shared_ptr<DynamicOps>CurOps)
{
	if (CheckPartialGradReady(BackwardOpsMap,OutputSetSize, CurOps))
	{
		if (OutputSetSize[CurOps.get()].empty())
		{
			GenEmptyGradDynamicTensor(Loss);
		}
		else
		{
			std::vector<DynamicOps*>OutputList;
			for (auto Item : OutputSetSize[CurOps.get()])OutputList.push_back(Item);
			DynamicTensor ThisOpsGradRes = DynamicTensor(BackwardOpsMap[CurOps.get()][OutputList[0]]);
			for (size_t a = 1; a < OutputList.size(); a++)
			{
				DynamicTensor PartRes = DynamicTensor(BackwardOpsMap[CurOps.get()][OutputList[a]]);
				ThisOpsGradRes = DynamicTensor::DynamicStdOps_Forward_Add({ ThisOpsGradRes, PartRes },he(), true);
			}
			if (CurOps->GradOps != nullptr)ThisOpsGradRes.Ops->TensorPointer = std::shared_ptr<Tensor>(CurOps->GradOps->TensorPointer->Add(ThisOpsGradRes.Ops->TensorPointer.get()));
			CurOps->GradOps = ThisOpsGradRes.Ops;
		}
		for (size_t a = 0; a < CurOps->InputOpsList.size(); a++)
		{
			if (BackwardOpsMap.find(CurOps->InputOpsList[a].get()) == BackwardOpsMap.end())
			{
				BackwardOpsMap[CurOps->InputOpsList[a].get()] = {};
			}
		}
		if(BackwardOps.find(CurOps->DynamicOpsType)!=BackwardOps.end())BackwardOps[CurOps->DynamicOpsType](BackwardOpsMap,CurOps);//ͨ通过算子传输partial grad
		for (size_t a = 0; a < CurOps->InputOpsList.size(); a++)
		{
			if (!CurOps->InputOpsList[a]->RequiresGrad)continue;//不需要求导的不用dfs
			BackwardDFS(BackwardOpsMap, OutputSetSize, Loss, CurOps->InputOpsList[a]);
		}
	}
	else
	{
		return;
	}
}

bool DynamicTensor::CheckPartialGradReady(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>&BackwardOpsMap, std::map<DynamicOps*, std::set<DynamicOps*>>& OutputSetSize, std::shared_ptr<DynamicOps>CurOps)
{
	if (BackwardOpsMap.find(CurOps.get()) == BackwardOpsMap.end())
	{
		BackwardOpsMap[CurOps.get()] = {};
	}
	return BackwardOpsMap[CurOps.get()].size() == OutputSetSize[CurOps.get()].size();
}

void DynamicTensor::GenEmptyGradDynamicTensor(DynamicTensor Loss)
{
	Tensor* GradResTensor;
	if(Loss.Ops == nullptr)
	{
		GradResTensor = Ops->TensorPointer->Copy();
		GradResTensor->FillArray(1.);
	}
	else GradResTensor = Loss.Ops->TensorPointer->Copy();
	DynamicTensor DynamicTensorGrad(std::shared_ptr<Tensor>(GradResTensor), Ops->RequiresGrad);
	Ops->GradOps = DynamicTensorGrad.Ops;
}

void DynamicTensor::GetAllOutputSizeBeforeBackward(std::map<DynamicOps*, std::set<DynamicOps*>>& OutputSetSize, std::shared_ptr<DynamicOps>CurOps)
{
	if (OutputSetSize.find(CurOps.get()) != OutputSetSize.end())return;
	OutputSetSize[CurOps.get()] = CurOps->OutputOpsSet;
	for (size_t a = 0; a < CurOps->InputOpsList.size(); a++)
	{
		GetAllOutputSizeBeforeBackward(OutputSetSize, CurOps->InputOpsList[a]);
	}
}

DynamicTensor DynamicTensor::ViewAndBC(DynamicTensor ThisDT, DynamicTensor Other, DynamicTensor(*InputFun)(std::vector<DynamicTensor>, he, bool), bool IsMatmul)
{
	//View
	auto ViewFun = [](DynamicTensor L, DynamicTensor S, DynamicTensor(*View)(std::vector<DynamicTensor>, he, bool))
		{
			he ViewParams = he::NewDict();
			ViewParams["ViewDims"] = he::NewList();
			size_t ResiShape = L.Ops->TensorPointer->shape.size() - S.Ops->TensorPointer->shape.size();
			for (size_t a = 0; a < ResiShape; a++)ViewParams["ViewDims"].append(1);
			for (size_t a = 0; a < S.Ops->TensorPointer->shape.size(); a++)ViewParams["ViewDims"].append(int(S.Ops->TensorPointer->shape[a]));
			return View({ S }, ViewParams, true);
		};
	if (!IsMatmul)
	{
		if (ThisDT.Ops->TensorPointer->shape.size() < Other.Ops->TensorPointer->shape.size())ThisDT = ViewFun(Other, ThisDT, DynamicStdOps_Forward_View);
		if (ThisDT.Ops->TensorPointer->shape.size() > Other.Ops->TensorPointer->shape.size())Other = ViewFun(ThisDT, Other, DynamicStdOps_Forward_View);
	}
	else
	{
		if (ThisDT.Ops->TensorPointer->shape.size() < Other.Ops->TensorPointer->shape.size())ThisDT = ViewFun(Other, ThisDT, DynamicStdOps_Forward_View);
		else if (ThisDT.Ops->TensorPointer->shape.size() > Other.Ops->TensorPointer->shape.size())
		{
			if (Other.Ops->TensorPointer->shape.size() == 1)
			{
				he ViewParams = he::NewDict();
				ViewParams["ViewDims"] = he::NewList();
				size_t ResiShape = ThisDT.Ops->TensorPointer->shape.size() - Other.Ops->TensorPointer->shape.size() - 1;
				for (size_t a = 0; a < ResiShape; a++)ViewParams["ViewDims"].append(1);
				for (size_t a = 0; a < Other.Ops->TensorPointer->shape.size(); a++)ViewParams["ViewDims"].append(int(Other.Ops->TensorPointer->shape[a]));
				ViewParams["ViewDims"].append(1);
				Other = DynamicStdOps_Forward_View({ Other }, ViewParams, true);
			}
			else Other = ViewFun(ThisDT, Other, DynamicStdOps_Forward_View);
		}
		else
		{
			if (Other.Ops->TensorPointer->shape.size() == 1)
			{
				he ThisViewParams = he::NewDict();
				ThisViewParams["ViewDims"] = he::NewList();
				he OtherViewParams = he::NewDict();
				OtherViewParams["ViewDims"] = he::NewList();
				ThisViewParams["ViewDims"].append(1);
				ThisViewParams["ViewDims"].append(int(ThisDT.Ops->TensorPointer->shape[0]));
				OtherViewParams["ViewDims"].append(int(Other.Ops->TensorPointer->shape[0]));
				OtherViewParams["ViewDims"].append(1);
				ThisDT = DynamicStdOps_Forward_View({ ThisDT }, ThisViewParams, true);
				Other = DynamicStdOps_Forward_View({ Other }, OtherViewParams, true);
			}
		}
		if (Other.Ops->TensorPointer->shape.size() == 2)
		{
			he MatmulParams = he::NewDict();
			MatmulParams["is_input_1st_T"] = false;
			MatmulParams["is_input_2nd_T"] = false;
			return InputFun({ ThisDT, Other }, MatmulParams, true);
		}
	}
	//BC
	he BCParams = he::NewDict();
	BCParams["BroadCastToShape"] = he::NewList();
	int BCFlag = 0;
	for (size_t a = 0; a < ThisDT.Ops->TensorPointer->shape.size() - 2*IsMatmul; a++)
	{
		int ThisShapeNum = ThisDT.Ops->TensorPointer->shape[a];
		int OtherShapeNum = Other.Ops->TensorPointer->shape[a];
		Log::Assert(!(ThisShapeNum != OtherShapeNum && std::min(ThisShapeNum, OtherShapeNum) != 1), "DynamicTensor Opeator+ Shape Error, Dim Can Not BroadCast");
		BCParams["BroadCastToShape"].append(std::max(ThisShapeNum, OtherShapeNum));
		if (ThisShapeNum < OtherShapeNum)BCFlag |= 1;
		if (ThisShapeNum > OtherShapeNum)BCFlag |= 2;
	}
	if (!IsMatmul)
	{
		if (!BCFlag)return InputFun({ ThisDT, Other }, he(), true);
		if (BCFlag == 1)return InputFun({ DynamicStdOps_Forward_BroadCastTo({ThisDT}, BCParams,true), Other }, he(), true);
		if (BCFlag == 2)return InputFun({ ThisDT, DynamicStdOps_Forward_BroadCastTo({Other}, BCParams,true) }, he(), true);
		return InputFun({ DynamicStdOps_Forward_BroadCastTo({ThisDT}, BCParams,true), DynamicStdOps_Forward_BroadCastTo({Other},BCParams,true) }, he(), true);
	}
	else
	{
		he MatmulParams = he::NewDict();
		MatmulParams["is_input_1st_T"] = false;
		MatmulParams["is_input_2nd_T"] = false;
		if (!BCFlag)return InputFun({ ThisDT, Other }, MatmulParams, true);
		if (BCFlag == 1)
		{
			BCParams["BroadCastToShape"].append(int(ThisDT.Ops->TensorPointer->shape[ThisDT.Ops->TensorPointer->shape.size() - 2]));
			BCParams["BroadCastToShape"].append(int(ThisDT.Ops->TensorPointer->shape[ThisDT.Ops->TensorPointer->shape.size() - 1]));
			return InputFun({ DynamicStdOps_Forward_BroadCastTo({ThisDT}, BCParams,true), Other }, MatmulParams, true);
		}
		if (BCFlag == 2)
		{
			BCParams["BroadCastToShape"].append(int(Other.Ops->TensorPointer->shape[Other.Ops->TensorPointer->shape.size() - 2]));
			BCParams["BroadCastToShape"].append(int(Other.Ops->TensorPointer->shape[Other.Ops->TensorPointer->shape.size() - 1]));
			return InputFun({ ThisDT, DynamicStdOps_Forward_BroadCastTo({Other}, BCParams,true) }, MatmulParams, true);
		}
		he OtherBCParams = BCParams;
		BCParams["BroadCastToShape"].append(int(ThisDT.Ops->TensorPointer->shape[ThisDT.Ops->TensorPointer->shape.size() - 2]));
		BCParams["BroadCastToShape"].append(int(ThisDT.Ops->TensorPointer->shape[ThisDT.Ops->TensorPointer->shape.size() - 1]));
		OtherBCParams["BroadCastToShape"].append(int(Other.Ops->TensorPointer->shape[Other.Ops->TensorPointer->shape.size() - 2]));
		OtherBCParams["BroadCastToShape"].append(int(Other.Ops->TensorPointer->shape[Other.Ops->TensorPointer->shape.size() - 1]));
		return InputFun({ DynamicStdOps_Forward_BroadCastTo({ThisDT}, BCParams,true), DynamicStdOps_Forward_BroadCastTo({Other},OtherBCParams,true) }, MatmulParams, true);
	}
}

DynamicTensor DynamicTensor::operator+(DynamicTensor Other)
{
	if (Other.Ops.get() != Ops.get())return ViewAndBC(*this, Other, DynamicStdOps_Forward_Add, false);
	else return DynamicStdOps_Forward_Add({ *this, DynamicStdOps_Forward_Add({Other},he(),true) }, he(), true);
}
DynamicTensor DynamicTensor::operator+(float Other)
{
	DynamicTensor Scalar({ 1 }, 0, Ops->TensorPointer->GetDeviceNum());
	Scalar.Fill(Other);
	return operator+(Scalar);
}

DynamicTensor DynamicTensor::operator%(DynamicTensor Other)
{
	if (Other.Ops.get() != Ops.get())return ViewAndBC(*this, Other, DynamicStdOps_Forward_Matmul, true);
	else
	{
		he MatmulParams = he::NewDict();
		MatmulParams["is_input_1st_T"] = false;
		MatmulParams["is_input_2nd_T"] = false;
		return DynamicStdOps_Forward_Matmul({ *this, DynamicStdOps_Forward_Add({Other},he(),true) }, MatmulParams, true);
	}
}
DynamicTensor DynamicTensor::operator*(DynamicTensor Other)
{
	if (Other.Ops.get() != Ops.get())return ViewAndBC(*this, Other, DynamicStdOps_Forward_Elemul, false);
	else return DynamicStdOps_Forward_Elemul({ *this, DynamicStdOps_Forward_Add({Other},he(),true) }, he(), true);
}
DynamicTensor DynamicTensor::operator*(float Other)
{
	DynamicTensor Scalar({ 1 }, 0, Ops->TensorPointer->GetDeviceNum());
	Scalar.Fill(Other);
	return operator*(Scalar);
}
DynamicTensor DynamicTensor::operator-(DynamicTensor Other)
{
	DynamicTensor MinusOne({1}, 0, Other.Ops->TensorPointer->GetDeviceNum());
	MinusOne.Fill(-1);
	return operator+(MinusOne * Other);
}
DynamicTensor DynamicTensor::operator-(float Other)
{
	DynamicTensor Scalar({ 1 }, 0, Ops->TensorPointer->GetDeviceNum());
	Scalar.Fill(Other);
	return operator-(Scalar);
}

DynamicTensor DynamicTensor::Sum(std::vector<int>Dims, bool KeepDim)
{
	if (Dims.size() == 0)
	{
		for (size_t a = 0; a < Ops->TensorPointer->shape.size(); a++)
		{
			Dims.push_back(a);
		}
	}
	he SumParams = he::NewDict();
	SumParams["SumDims"] = he::NewList();
	for (size_t a = 0; a < Dims.size(); a++)SumParams["SumDims"].append(Dims[a]);
	DynamicTensor Res = DynamicStdOps_Forward_Sum({ *this }, SumParams, true);
	if (KeepDim)return Res;
	std::map<int, int>DimsMp;
	for (size_t a = 0; a < Dims.size(); a++)DimsMp[Dims[a]] = 1;
	he ViewParams = he::NewDict();
	ViewParams["ViewDims"] = he::NewList();
	for (size_t a = 0; a < Res.Ops->TensorPointer->shape.size(); a++)
	{
		if (DimsMp.find(int(a)) == DimsMp.end())ViewParams["ViewDims"].append(int(Res.Ops->TensorPointer->shape[a]));
		else continue;
	}
	return DynamicStdOps_Forward_View({ Res }, ViewParams, true);
}

DynamicTensor DynamicTensor::View(std::vector<int>Dims)
{
	he ViewParams = he::NewDict();
	ViewParams["ViewDims"] = he::NewList();
	for (size_t a = 0; a < Dims.size(); a++)ViewParams["ViewDims"].append(Dims[a]);
	return DynamicTensor::DynamicStdOps_Forward_View({ *this }, ViewParams, true);
}

DynamicTensor DynamicTensor::Softmax(int InputDim)
{
	he SoftmaxParams = he::NewDict();
	if(InputDim<0)InputDim = Shape().size()+InputDim;
	SoftmaxParams["SoftmaxDim"] = InputDim;
	return DynamicStdOps_Forward_Softmax({ *this }, SoftmaxParams, true);
}

DynamicTensor DynamicTensor::Pow(float EleExponent)
{
	he PowParams = he::NewDict();
	PowParams["EleExponent"] = EleExponent;
	return DynamicStdOps_Forward_Pow({ *this }, PowParams, true);
}

DynamicTensor DynamicTensor::Dropout(DynamicTensor Input, float P, bool InPlace)
{
	if (Input.Ops->IsEval)return Input;
	auto DropoutTensor = Input.Ops->TensorPointer->Copy();
	DropoutTensor->FillRandomValBernoulli(1-P);
	Tensor* DropoutTensorDotP = DropoutTensor->MulScalar(1 / (1-P));
	delete DropoutTensor;
	Input.Ops->TensorPointer = std::shared_ptr<Tensor>(DropoutTensorDotP->EleMul(Input.Ops->TensorPointer.get()));
	delete DropoutTensorDotP;
	return Input;
}

std::vector<DynamicTensor> DynamicTensor::Split(int SplitSize, int Dim)
{
	std::vector<int>SplitSections;
	int ProtoDimSize = Ops->TensorPointer->shape[Dim];
	while(ProtoDimSize)
	{
		if(ProtoDimSize > SplitSize)
		{
			SplitSections.push_back(SplitSize);
			ProtoDimSize -= SplitSize;
		}
		else
		{
			SplitSections.push_back(ProtoDimSize);
			ProtoDimSize = 0;
		}
	}
	return Split(SplitSections, Dim);
}
std::vector<DynamicTensor> DynamicTensor::Split(std::vector<int> SplitSections, int Dim)
{
	auto GenLeftMul = Ops->TensorPointer->GenerateSplitTensor(SplitSections, Dim);
	std::vector<DynamicTensor>Res;
	int PreDims = 1;
	int LastDims = 1;
	for (size_t a = 0; a < Ops->TensorPointer->shape.size(); a++)
	{
		if (a < Dim)PreDims *= Ops->TensorPointer->shape[a];
		else LastDims *= Ops->TensorPointer->shape[a];
	}
	DynamicTensor ViewTensor = View({ PreDims , LastDims });
	for (size_t a = 0; a < GenLeftMul.size(); a++)
	{
		DynamicTensor ResTMPTensor = ViewTensor % DynamicTensor(std::shared_ptr<Tensor>(GenLeftMul[a]));
		std::vector<int> ReturnShape;
		for (size_t b = 0; b < Ops->TensorPointer->shape.size(); b++)
		{
			if (b != Dim)ReturnShape.push_back(Ops->TensorPointer->shape[b]);
			else ReturnShape.push_back(SplitSections[a]);
		}
		Res.push_back(ResTMPTensor.View(ReturnShape));
	}
	return Res;
}

DynamicTensor DynamicTensor::Eleexp(float EleBaseNum)
{
	he EleexpParams = he::NewDict();
	EleexpParams["EleBaseNum"] = EleBaseNum;
	return DynamicStdOps_Forward_Eleexp({ *this }, EleexpParams, true);
}

DynamicTensor DynamicTensor::Tanh()
{
	DynamicTensor ExpTMP =  (DynamicTensor(Ops) * (-2)).Eleexp(M_E);
	return (ExpTMP * (-1) + 1) * ((ExpTMP + 1.).Pow(-1.));
}

DynamicTensor DynamicTensor::Cat(std::vector<DynamicTensor>InputTensors, int Dim)
{
	int InputNum = InputTensors.size();
	std::vector<int> ReturnShape,StartShape,EndShape;
	for (size_t a = 0; a < InputTensors[0].Shape().size(); a++)
	{
		// 这里的修改，我们认为InputTensors中除了Dim所在的维度其他shape都是相等的
		ReturnShape.push_back(InputTensors[0].Shape()[a]);
		StartShape.push_back(0);
		EndShape.push_back(InputTensors[0].Shape()[a] - 1);
	}
	he SubSendParams = he::NewDict();
	int TargetDim = 0;
	SubSendParams["InputStartShape"] = he::NewList(InputNum);
    SubSendParams["SubInputShapeS"] = he::NewList(InputNum);
    SubSendParams["SubInputShapeE"] = he::NewList(InputNum);
	for (int a = 0; a < InputTensors.size(); a++)
	{
		auto ThisStartShape = StartShape;
		ThisStartShape[Dim] = TargetDim;
		SubSendParams["InputStartShape"][a] = he::NewList(ThisStartShape);
		SubSendParams["SubInputShapeS"][a] = he::NewList(StartShape);
		auto ThisEndShape = EndShape;
		ThisEndShape[Dim] = InputTensors[a].Shape()[Dim]-1;
		SubSendParams["SubInputShapeE"][a] = he::NewList(ThisEndShape);
		TargetDim += InputTensors[a].Shape()[Dim];
	}
	ReturnShape[Dim] = TargetDim;
	SubSendParams["TargetShape"] = he::NewList(ReturnShape);
	return DynamicStdOps_Forward_SubSend(InputTensors, SubSendParams, true);
}

DynamicTensor DynamicTensor::GELU()
{
	auto Self = DynamicTensor(Ops);
	return (Self * 0.5) * (((Self + Self.Pow(3.) * 0.044715) * std::pow(2. / M_PI, 0.5)).Tanh() + 1);
}

DynamicTensor DynamicTensor::Mean(std::vector<int>InputDims, bool KeepDim)
{
	float MeanPartial = 1;
	if(InputDims.empty())
	{
		for (size_t a = 0; a < Shape().size(); a++)MeanPartial *= Shape()[a];
	}
	else
	{
		for (size_t a = 0; a < InputDims.size(); a++)MeanPartial *= Shape()[InputDims[a]];
	}
	return Sum(InputDims, KeepDim)*(1./ MeanPartial);
}

DynamicTensor DynamicTensor::Var(std::vector<int>InputDims, bool KeepDim, float Correction)
{
	auto Self = DynamicTensor(Ops);
	float SumDimRes = 1;
	for (size_t a = 0; a < InputDims.size(); a++)SumDimRes *= Ops->TensorPointer->shape[InputDims[a]];
	return (Self - Self.Mean(InputDims, true)).Pow(2.).Sum(InputDims, KeepDim) * (1. / (SumDimRes - Correction));
}

DynamicTensor DynamicTensor::Tril(int Diagonal)
{
	auto Self = DynamicTensor(Ops);
	Tensor* TensorTrilOnes = Tensor::GenerateTrilOnes(Shape(), Diagonal, Ops->TensorPointer->GetDeviceNum());
	return Self*DynamicTensor(std::shared_ptr<Tensor>(TensorTrilOnes));
}

DynamicTensor DynamicTensor::Transpose(int Dim0, int Dim1, int DebugFlag)
{
	he TranposeParams = he::NewDict();
	if(Dim0<0)Dim0 = Ops->TensorPointer->shape.size()+Dim0;
	if(Dim1<0)Dim1 = Ops->TensorPointer->shape.size()+Dim1;
	TranposeParams["Dim0"] = Dim0;
	TranposeParams["Dim1"] = Dim1;
	return DynamicStdOps_Forward_Transpose({*this}, TranposeParams, true);
}

DynamicTensor DynamicTensor::MaskedFill(DynamicTensor Mask, float Value)
{
	DynamicTensor Ones({1}, 0, Mask.Ops->TensorPointer->GetDeviceNum());
	Ones.Fill(1);
	return DynamicTensor(Ops)*(Ones - Mask)+Mask*Value;
}

DynamicTensor DynamicTensor::EleLog()
{
	auto Self = DynamicTensor(Ops);
	return DynamicStdOps_Forward_EleLog({Self}, he(), true);
}

DynamicTensor DynamicTensor::CrossEntropy(DynamicTensor Input, DynamicTensor Target, std::string Reduction, DynamicTensor Weight, float LabelSmoothing)
{
	if(Weight.Ops == nullptr)
	{
		Weight = DynamicTensor(Input.Shape(), false ,Input.GetDeviceNum());
		Weight.Fill(1.);
	}
	auto ExpTensor = Input.Eleexp(M_E);
	auto ExpTensorSum = ExpTensor.Sum({1}, true).Pow(-1);
	auto MiniBatchRes = (ExpTensor*ExpTensorSum).EleLog()*Target*Weight;
	auto MiniBatchResSum = MiniBatchRes.Sum()*(-1);
	if(Reduction == "Sum")return MiniBatchResSum;
	else return MiniBatchResSum*(1./Input.Shape()[0]);
}

DynamicTensor DynamicTensor::Sigmoid()
{
	auto Self = DynamicTensor(Ops);
	return  ((Self*(-1)).Eleexp(M_E)+ 1).Pow(-1.);
}

DynamicTensor DynamicTensor::GaussianCdf(float InputMean, float InputStd, int Terms)
{
	auto Self = DynamicTensor(Ops);
	DynamicTensor ErfApprox = Self*0;
	for(int a=0;a<Terms;a++)
	{
		float Coe = Factorial(a)*(2*a+1);
		if(a%2)Coe = -1/Coe;
		else Coe = 1/Coe;
		ErfApprox = ErfApprox+Self.Pow(2*a+1)*Coe;
	}
	DynamicTensor CDF = ErfApprox*(1./std::sqrt(M_PI)) + 0.5;
	return CDF;
}

DynamicTensor DynamicTensor::ReLU()
{
	auto Self = DynamicTensor(Ops);
	Tensor* SignTensor = Self.Ops->TensorPointer->GenerateSignTensor();
	return Self*DynamicTensor(std::shared_ptr<Tensor>(SignTensor));
}

DynamicTensor DynamicTensor::Abs()
{
    DynamicTensor MinusSelf = (DynamicTensor(Ops)*(-1)).ReLU();
    DynamicTensor Self = DynamicTensor(Ops).ReLU();
    return MinusSelf + Self;
}

DynamicTensor DynamicTensor::Cholesky()
{
    typedef typename std::decay<decltype(*(Ops->TensorPointer))>::type ContentType;
    auto CholeskyRes = std::shared_ptr<ContentType>(Ops->TensorPointer->Cholesky());
    return DynamicTensor(CholeskyRes, Ops->RequiresGrad);
}

DynamicTensor DynamicTensor::SampleFromStdGaussian(int Dim, std::vector<int> InputVec, int Seed,int DeviceNum)
{
    std::vector<size_t> ShapeVec;
    for(auto&it:InputVec)ShapeVec.push_back(it);
    if(Seed == -1)Seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto ContentRes = Tensor::SampleMultivariateStandardGaussian(Dim, ShapeVec, Seed, DeviceNum);
    using ContentType = typename std::remove_pointer<typename std::decay<decltype(ContentRes)>::type>::type;
    auto ContentPtr = std::shared_ptr<ContentType>(ContentRes);
    return DynamicTensor(ContentPtr);
}

DynamicTensor DynamicTensor::SampleFromOtherGaussian(int Dim, std::vector<int> InputVec, DynamicTensor Mean, DynamicTensor Var,DynamicTensor VarL, int Seed,int DeviceNum)
{
    if(Seed == -1)Seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto GaussianShape = Mean.Shape();
    for(size_t a = 0;a+1<GaussianShape.size();a++)InputVec.push_back(GaussianShape[a]);
    auto OutputShape = InputVec;
    OutputShape.push_back(Dim);
    InputVec.push_back(1);
    DynamicTensor STDSamples = DynamicTensor::SampleFromStdGaussian(Dim, InputVec, Seed, DeviceNum); //(10,2,3)
    if(VarL.Ops == nullptr)VarL= Var.Cholesky();
    return Mean + (STDSamples%VarL.Transpose(-1,-2)).View(OutputShape);// mean:(2,3), varl:(2,3,3)
}

DynamicTensor DynamicTensor::Inverse()
{
    auto InverseContent = Ops->TensorPointer->Inverse();
    using ContentType = typename std::remove_pointer<typename std::decay<decltype(InverseContent)>::type>::type;
    auto ContentPtr = std::shared_ptr<ContentType>(InverseContent);
    return DynamicTensor(ContentPtr);
}

DynamicTensor DynamicTensor::Det_Symmetric(DynamicTensor InputL)
{
    DynamicTensor UnitTensor = DynamicTensor::CreateUnitTensor(InputL.ShapeInt(), InputL.Ops->RequiresGrad, InputL.GetDeviceNum());
    DynamicTensor AllOnes = InputL.Copy();
    AllOnes.Fill(1);
    DynamicTensor DiagRes = InputL*UnitTensor + AllOnes - UnitTensor;
    int ShapeLen = InputL.Shape().size();
    DynamicTensor Res = DiagRes.EleLog().Sum({ShapeLen-2, ShapeLen-1}, true).Eleexp(M_E).Pow(2.);
    return Res;
}

DynamicTensor DynamicTensor::ProbabilityDensity_Gaussian(DynamicTensor InputSample, DynamicTensor InputMean, DynamicTensor InputVarInv, DynamicTensor InputVarDet)
{
    // sample_num:m, gaussian_num:u, dim_num:n
    // InputSample:(m, u, n)
    // InputMean:(u, n)
    // InputVarInv:(u, n, n)
    // InputVarDet:(u, 1, 1)
    auto OutputShape = InputSample.ShapeInt();
    OutputShape.push_back(1);
    DynamicTensor XMinusMean = (InputSample - InputMean).View(OutputShape); //(m, u, n, 1)
    DynamicTensor XMinusMeanT = XMinusMean.Transpose(-1, -2); //(m, u, 1, n)
    DynamicTensor ExpPartial = (XMinusMeanT % InputVarInv % XMinusMean * (-0.5)).Eleexp(M_E); //(m, u, 1, 1)
    DynamicTensor CPartial = InputVarDet.Pow(-0.5)*std::pow(2.*M_PI, -InputSample.ShapeInt().back()*0.5); //(m, u, 1, 1)
    DynamicTensor ProtoRes = ExpPartial*CPartial; //(m, u, 1, 1)
    auto ProtoShape = ProtoRes.ShapeInt(); //(m, u, 1, 1)
    ProtoShape.pop_back(); //(m, u, 1)
    ProtoShape.pop_back(); //(m, u)
    DynamicTensor FinalRes = ProtoRes.View(ProtoShape);//(m, u)
    return FinalRes;
}

DynamicTensor DynamicTensor::ProbabilityDensity_Log_Gaussian(DynamicTensor InputSample, DynamicTensor InputMean, DynamicTensor InputVarInv, DynamicTensor InputVarDet)
{
    // sample_num:m, gaussian_num:u, dim_num:n
    // InputSample:(m, u, n)
    // InputMean:(u, n)
    // InputVarInv:(u, n, n)
    // InputVarDet:(u, 1, 1)
    auto OutputShape = InputSample.ShapeInt();
    OutputShape.push_back(1);
    DynamicTensor XMinusMean = (InputSample - InputMean).View(OutputShape); //(m, u, n, 1)
    DynamicTensor XMinusMeanT = XMinusMean.Transpose(-1, -2); //(m, u, 1, n)
    DynamicTensor LogExpPartial = XMinusMeanT % InputVarInv % XMinusMean * (-0.5); //(m, u, 1, 1)
    DynamicTensor LogCPartial = InputVarDet.EleLog()*(-0.5) - InputSample.ShapeInt().back()*0.5*std::log(2.*M_PI) ; //(m, u, 1, 1)
    DynamicTensor ProtoRes = LogExpPartial + LogCPartial; //(m, u, 1, 1)
    auto ProtoShape = ProtoRes.ShapeInt(); //(m, u, 1, 1)
    ProtoShape.pop_back(); //(m, u, 1)
    ProtoShape.pop_back(); //(m, u)
    DynamicTensor FinalRes = ProtoRes.View(ProtoShape);//(m, u)
    return FinalRes;
}

DynamicTensor DynamicTensor::Max(std::vector<int>Dims, bool KeepDim)
{
    if (Dims.size() == 0)
	{
		for (size_t a = 0; a < Ops->TensorPointer->shape.size(); a++)
		{
			Dims.push_back(a);
		}
	}
    Tensor* Res = Ops->TensorPointer->MaxOrMin(Dims, true);
    DynamicTensor ReturnRes = DynamicTensor(std::shared_ptr<Tensor>(Res), false);
    if(KeepDim)return ReturnRes;
    else
    {
        auto OutputPreShape = ShapeInt();
        for(auto&thisDim:Dims)OutputPreShape[thisDim] = -1;
        std::vector<size_t> OutputShape;
        for(auto&thisDim:OutputPreShape)
        {
            if(thisDim > 0)
            {
                OutputShape.push_back(thisDim);
            }
        }
        ReturnRes.Ops->TensorPointer->shape = OutputShape;
        return ReturnRes;
    }
}

void DynamicTensor::OpsSetInMap()
{
	BackwardOps[OpsType::Add] = DynamicStdOps_Backward_Add;
	BackwardOps[OpsType::MatMul] = DynamicStdOps_Backward_Matmul;
	BackwardOps[OpsType::Sum] = DynamicStdOps_Backward_Sum;
	BackwardOps[OpsType::BroadCastTo] = DynamicStdOps_Backward_BroadCastTo;
	BackwardOps[OpsType::View] = DynamicStdOps_Backward_View;
	BackwardOps[OpsType::EleMul] = DynamicStdOps_Backward_Elemul;
	BackwardOps[OpsType::Softmax] = DynamicStdOps_Backward_Softmax;
	BackwardOps[OpsType::Pow] = DynamicStdOps_Backward_Pow;
	BackwardOps[OpsType::EleExp] = DynamicStdOps_Backward_Eleexp;
	BackwardOps[OpsType::Transpose] = DynamicStdOps_Backward_Transpose;
	BackwardOps[OpsType::EleLog] = DynamicStdOps_Backward_EleLog;
	BackwardOps[OpsType::SubSend] = DynamicStdOps_Backward_SubSend;
}


DynamicTensor DynamicTensor::DynamicStdOps_Forward_Add(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	auto ResTensorContent = InputList[0].Ops->TensorPointer->Copy();
	for (size_t a = 1; a < InputList.size(); a++)
	{
		Tensor* TMPTensor = ResTensorContent->Add(InputList[a].Ops->TensorPointer.get());
		delete ResTensorContent;
		ResTensorContent = TMPTensor;
	}
	return SetComputationalHistory(ResTensorContent, InputList, InputParams, OpsType::Add, RequiresGrad);
}
void DynamicTensor::DynamicStdOps_Backward_Add(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	for (size_t a = 0; a < CurOps->InputOpsList.size(); a++)
	{
		if (!CurOps->InputOpsList[a]->RequiresGrad)continue;
		auto AddRes = DynamicTensor::DynamicStdOps_Forward_Add({ DynamicTensor(CurOps->GradOps) }, he(), true);
		BackwardOpsMap[CurOps->InputOpsList[a].get()][CurOps.get()] = AddRes.Ops;
	}
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_Matmul(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	bool IsInput1stT = InputParams["is_input_1st_T"].i();
	bool IsInput2ndT = InputParams["is_input_2nd_T"].i();
	Tensor* Input1stRes, * Input2ndRes;
	if (IsInput1stT)Input1stRes = InputList[0].Ops->TensorPointer->T();
	else Input1stRes = InputList[0].Ops->TensorPointer.get();
	if (IsInput2ndT)Input2ndRes = InputList[1].Ops->TensorPointer->T();
	else Input2ndRes = InputList[1].Ops->TensorPointer.get();
	auto TensorResult = Input1stRes->Matmul(Input2ndRes);
	if (IsInput1stT)delete Input1stRes;
	if (IsInput2ndT)delete Input2ndRes;
	return SetComputationalHistory(TensorResult, InputList, InputParams, OpsType::MatMul, RequiresGrad);
}
void DynamicTensor::DynamicStdOps_Backward_Matmul(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>&BackwardOpsMap,std::shared_ptr<DynamicOps>CurOps)
{
	bool IsInput1stT = CurOps->Params["is_input_1st_T"].i();
	bool IsInput2ndT = CurOps->Params["is_input_2nd_T"].i();
	if (CurOps->InputOpsList[0]->RequiresGrad)
	{
		he GradParams = he::NewDict();
		DynamicTensor TensorRes;
		if (IsInput1stT == false && IsInput2ndT == false)
		{
			GradParams["is_input_1st_T"] = false;
			GradParams["is_input_2nd_T"] = true;
			TensorRes = DynamicStdOps_Forward_Matmul({ DynamicTensor(CurOps->GradOps),DynamicTensor(CurOps->InputOpsList[1]) }, GradParams, true);
		}
		if (IsInput1stT == false && IsInput2ndT == true)
		{
			GradParams["is_input_1st_T"] = false;
			GradParams["is_input_2nd_T"] = false;
			TensorRes = DynamicStdOps_Forward_Matmul({ DynamicTensor(CurOps->GradOps),DynamicTensor(CurOps->InputOpsList[1]) }, GradParams, true);
		}
		if (IsInput1stT == true && IsInput2ndT == false)
		{
			GradParams["is_input_1st_T"] = false;
			GradParams["is_input_2nd_T"] = true;
			TensorRes = DynamicStdOps_Forward_Matmul({ DynamicTensor(CurOps->InputOpsList[1]),DynamicTensor(CurOps->GradOps) }, GradParams, true);
		}
		if (IsInput1stT == true && IsInput2ndT == true)
		{
			GradParams["is_input_1st_T"] = true;
			GradParams["is_input_2nd_T"] = true;
			TensorRes = DynamicStdOps_Forward_Matmul({ DynamicTensor(CurOps->InputOpsList[1]),DynamicTensor(CurOps->GradOps) }, GradParams, true);
		}
		BackwardOpsMap[CurOps->InputOpsList[0].get()][CurOps.get()] = TensorRes.Ops;
	}
	if (CurOps->InputOpsList[1]->RequiresGrad)
	{
		he GradParams = he::NewDict();
		DynamicTensor TensorRes;
		if (IsInput1stT == false && IsInput2ndT == false)
		{
			GradParams["is_input_1st_T"] = true;
			GradParams["is_input_2nd_T"] = false;
			TensorRes = DynamicStdOps_Forward_Matmul({ DynamicTensor(CurOps->InputOpsList[0]),DynamicTensor(CurOps->GradOps) }, GradParams, true);
		}
		if (IsInput1stT == false && IsInput2ndT == true)
		{
			GradParams["is_input_1st_T"] = true;
			GradParams["is_input_2nd_T"] = false;
			TensorRes = DynamicStdOps_Forward_Matmul({ DynamicTensor(CurOps->GradOps),DynamicTensor(CurOps->InputOpsList[0]) }, GradParams, true);
		}
		if (IsInput1stT == true && IsInput2ndT == false)
		{
			GradParams["is_input_1st_T"] = false;
			GradParams["is_input_2nd_T"] = false;
			TensorRes = DynamicStdOps_Forward_Matmul({ DynamicTensor(CurOps->InputOpsList[0]),DynamicTensor(CurOps->GradOps) }, GradParams, true);
		}
		if (IsInput1stT == true && IsInput2ndT == true)
		{
			GradParams["is_input_1st_T"] = true;
			GradParams["is_input_2nd_T"] = true;
			TensorRes = DynamicStdOps_Forward_Matmul({ DynamicTensor(CurOps->GradOps),DynamicTensor(CurOps->InputOpsList[0]) }, GradParams, true);
		}
		BackwardOpsMap[CurOps->InputOpsList[1].get()][CurOps.get()] = TensorRes.Ops;
	}
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_BroadCastTo(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	std::vector<size_t>BroadCastToShape;
	for (he a = 0; a < InputParams["BroadCastToShape"].size(); a = a + 1)BroadCastToShape.push_back(InputParams["BroadCastToShape"][a].i());
	auto TensorResult = InputList[0].Ops->TensorPointer->BroadCastTo(BroadCastToShape);
	return SetComputationalHistory(TensorResult, InputList, InputParams, OpsType::BroadCastTo, RequiresGrad);
}
void DynamicTensor::DynamicStdOps_Backward_BroadCastTo(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	if (!CurOps->InputOpsList[0]->RequiresGrad)return;
	std::vector<size_t>BroadCastToShape;
	for (he a = 0; a < CurOps->Params["BroadCastToShape"].size(); a = a + 1)BroadCastToShape.push_back(CurOps->Params["BroadCastToShape"][a].i());
	he InputParams = he::NewDict();
	InputParams["SumDims"] = he::NewList();
	for (size_t a = 0; a < CurOps->TensorPointer->shape.size(); a++)
	{
		if (CurOps->TensorPointer->shape[a] != CurOps->InputOpsList[0]->TensorPointer->shape[a])
		{
			InputParams["SumDims"].append(int(a));
		}
	}
	DynamicTensor DynamicTensorRes = DynamicStdOps_Forward_Sum({ DynamicTensor(CurOps->GradOps) }, InputParams, true);
	BackwardOpsMap[CurOps->InputOpsList[0].get()][CurOps.get()] = DynamicTensorRes.Ops;
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_Sum(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	std::vector<size_t>SumDims;
	for (he a = 0; a < InputParams["SumDims"].size(); a = a + 1)SumDims.push_back(InputParams["SumDims"][a].i());
	auto TensorResult = InputList[0].Ops->TensorPointer->Sum(SumDims);
	return SetComputationalHistory(TensorResult, InputList, InputParams, OpsType::Sum, RequiresGrad);
}
void DynamicTensor::DynamicStdOps_Backward_Sum(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	if (!CurOps->InputOpsList[0]->RequiresGrad)return;
	he InputParams = he::NewDict();
	InputParams["BroadCastToShape"] = he::NewList();
	for (size_t a = 0; a < CurOps->InputOpsList[0]->TensorPointer->shape.size(); a++)InputParams["BroadCastToShape"].append(int(CurOps->InputOpsList[0]->TensorPointer->shape[a]));
	DynamicTensor DynamicTensorRes = DynamicStdOps_Forward_BroadCastTo({ DynamicTensor(CurOps->GradOps) }, InputParams, true);
	BackwardOpsMap[CurOps->InputOpsList[0].get()][CurOps.get()] = DynamicTensorRes.Ops;
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_View(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	std::vector<size_t>ViewDims;
	int MinusIndex = -1;
	for (he a = 0; a < InputParams["ViewDims"].size(); a = a + 1)
	{
		if (InputParams["ViewDims"][a] < 0)
		{
			MinusIndex = a.i();
			ViewDims.push_back(0);
		}
		else ViewDims.push_back(InputParams["ViewDims"][a].i());
	}
	auto TensorResult = InputList[0].Ops->TensorPointer->View(ViewDims,MinusIndex);
	return SetComputationalHistory(TensorResult, InputList, InputParams, OpsType::View, RequiresGrad);
}
void DynamicTensor::DynamicStdOps_Backward_View(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	if (!CurOps->InputOpsList[0]->RequiresGrad)return;
	he InputParams = he::NewDict();
	InputParams["ViewDims"] = he::NewList();
	for (size_t a = 0; a < CurOps->InputOpsList[0]->TensorPointer->shape.size(); a++)InputParams["ViewDims"].append(int(CurOps->InputOpsList[0]->TensorPointer->shape[a]));
	DynamicTensor DynamicTensorRes = DynamicStdOps_Forward_View({ DynamicTensor(CurOps->GradOps) }, InputParams, true);
	BackwardOpsMap[CurOps->InputOpsList[0].get()][CurOps.get()] = DynamicTensorRes.Ops;
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_Elemul(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	auto ResTensorContent = InputList[0].Ops->TensorPointer->EleMul(InputList[1].Ops->TensorPointer.get());
	return SetComputationalHistory(ResTensorContent, InputList, InputParams, OpsType::EleMul, RequiresGrad);
}
void DynamicTensor::DynamicStdOps_Backward_Elemul(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	if (CurOps->InputOpsList[0]->RequiresGrad)
	{
		DynamicTensor DynamicTensorRes = DynamicStdOps_Forward_Elemul({ DynamicTensor(CurOps->GradOps), DynamicTensor(CurOps->InputOpsList[1]) }, he(), true);
		BackwardOpsMap[CurOps->InputOpsList[0].get()][CurOps.get()] = DynamicTensorRes.Ops;
	}
	if (CurOps->InputOpsList[1]->RequiresGrad)
	{
		DynamicTensor DynamicTensorRes = DynamicStdOps_Forward_Elemul({ DynamicTensor(CurOps->GradOps), DynamicTensor(CurOps->InputOpsList[0]) }, he(), true);
		BackwardOpsMap[CurOps->InputOpsList[1].get()][CurOps.get()] = DynamicTensorRes.Ops;
	}
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_Softmax(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	int SoftmaxDim = InputParams["SoftmaxDim"].i();
	auto ResTensorContent = InputList[0].Ops->TensorPointer->Softmax(SoftmaxDim);
	return SetComputationalHistory(ResTensorContent, InputList, InputParams, OpsType::Softmax, RequiresGrad);
}
void DynamicTensor::DynamicStdOps_Backward_Softmax(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	/**
	return DynamicTensor::D = {d_i, i \in [1,n]}
	any d_i = I_i*(O_i - I \dot O)
	*/
	if (!CurOps->InputOpsList[0]->RequiresGrad)return;
	int SoftmaxDim = CurOps->Params["SoftmaxDim"].i();
	DynamicTensor MinusRes = (DynamicTensor(CurOps) * DynamicTensor(CurOps->GradOps)).Sum({ SoftmaxDim },true);
	DynamicTensor Res = DynamicTensor(CurOps) * (DynamicTensor(CurOps->GradOps) - MinusRes);
	BackwardOpsMap[CurOps->InputOpsList[0].get()][CurOps.get()] = Res.Ops;
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_Pow(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	float EleExponent = InputParams["EleExponent"].f();
	auto ResTensorContent = InputList[0].Ops->TensorPointer->Pow(EleExponent);
	return SetComputationalHistory(ResTensorContent, InputList, InputParams, OpsType::Pow, RequiresGrad);
}
void DynamicTensor::DynamicStdOps_Backward_Pow(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	if (!CurOps->InputOpsList[0]->RequiresGrad)return;
	float CurEleExponent = CurOps->Params["EleExponent"].f();
	he PowParams = he::NewDict();
	PowParams["EleExponent"] = CurEleExponent - 1;
	DynamicTensor Res = DynamicStdOps_Forward_Pow({ DynamicTensor(CurOps->InputOpsList[0])}, PowParams, true);
	Res = DynamicTensor(CurOps->GradOps)*Res * CurEleExponent;
	BackwardOpsMap[CurOps->InputOpsList[0].get()][CurOps.get()] = Res.Ops;
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_Eleexp(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	float EleBaseNum = InputParams["EleBaseNum"].f();
	auto ResTensorContent = InputList[0].Ops->TensorPointer->EleExp(EleBaseNum);
	return SetComputationalHistory(ResTensorContent, InputList, InputParams, OpsType::EleExp, RequiresGrad);
}
void DynamicTensor::DynamicStdOps_Backward_Eleexp(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	if (!CurOps->InputOpsList[0]->RequiresGrad)return;
	float EleBaseNum = CurOps->Params["EleBaseNum"].f();
	DynamicTensor Res = DynamicTensor(CurOps) * DynamicTensor(CurOps->GradOps) * std::log(EleBaseNum);
	BackwardOpsMap[CurOps->InputOpsList[0].get()][CurOps.get()] = Res.Ops;
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_Transpose(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	int Dim0 = InputParams["Dim0"].i();
	int Dim1 = InputParams["Dim1"].i();
	auto ResTensorContent = InputList[0].Ops->TensorPointer->Transpose(Dim0, Dim1);
	return SetComputationalHistory(ResTensorContent, InputList, InputParams, OpsType::Transpose, RequiresGrad); 
}
void DynamicTensor::DynamicStdOps_Backward_Transpose(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	if (!CurOps->InputOpsList[0]->RequiresGrad)return;
	int Dim0 = CurOps->Params["Dim0"].i();
	int Dim1 = CurOps->Params["Dim1"].i();
	DynamicTensor Res = DynamicTensor(CurOps->GradOps).Transpose(Dim0, Dim1);
	BackwardOpsMap[CurOps->InputOpsList[0].get()][CurOps.get()] = Res.Ops;
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_EleLog(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	auto ResTensorContent = InputList[0].Ops->TensorPointer->EleLog();
	return SetComputationalHistory(ResTensorContent, InputList, InputParams, OpsType::EleLog, RequiresGrad); 
}
void DynamicTensor::DynamicStdOps_Backward_EleLog(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	if (!CurOps->InputOpsList[0]->RequiresGrad)return;
	DynamicTensor Res = DynamicTensor(CurOps->GradOps)*DynamicTensor(CurOps->InputOpsList[0]).Pow(-1);
	BackwardOpsMap[CurOps->InputOpsList[0].get()][CurOps.get()] = Res.Ops;
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_SubSend(std::vector<DynamicTensor>InputList, he InputParams, bool RequiresGrad)
{
	std::vector<size_t>TargetShape;
	InputParams["TargetShape"].v(TargetShape);
	auto ResTensorContent = new Tensor(TargetShape, InputList[0].Ops->TensorPointer->GetDeviceNum());
	ResTensorContent->FillArray(0);
	for(int a=0;a<InputList.size();a++)
	{
		std::vector<size_t>InputStartShape;
		InputParams["InputStartShape"][a].v(InputStartShape);
		std::vector<size_t>SubInputShapeS;
		InputParams["SubInputShapeS"][a].v(SubInputShapeS);
		std::vector<size_t>SubInputShapeE;
		InputParams["SubInputShapeE"][a].v(SubInputShapeE);
		auto SubInputTensor = InputList[a].Ops->TensorPointer->GetTensorBy2ShapeVector(SubInputShapeS, SubInputShapeE);
		SubInputTensor->SendTensorBy2ShapeVector(InputStartShape, ResTensorContent);
		delete SubInputTensor;
	}
	return SetComputationalHistory(ResTensorContent, InputList, InputParams, OpsType::SubSend, RequiresGrad); 
}
void DynamicTensor::DynamicStdOps_Backward_SubSend(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	std::vector<int>ShapeVec;
	auto& P = CurOps->Params;
	for(int a=0;a<CurOps->InputOpsList.size();a++)
	{
		auto& ThisInput = CurOps->InputOpsList[a];
		if (!ThisInput->RequiresGrad)continue;
		he InputTensorParams = he::NewDict();
		InputTensorParams["InputStartShape"] = he::NewList(1);
    	InputTensorParams["SubInputShapeS"] = he::NewList(1);
    	InputTensorParams["SubInputShapeE"] = he::NewList(1);
		ShapeVec.clear();
		for(auto&it:ThisInput->TensorPointer->shape)ShapeVec.push_back(it);
		InputTensorParams["TargetShape"] = he::NewList(ShapeVec);
		P["InputStartShape"][a].v(ShapeVec);
		InputTensorParams["SubInputShapeS"][0] = he::NewList(ShapeVec);
		P["SubInputShapeS"][a].v(ShapeVec);
		InputTensorParams["InputStartShape"][0] = he::NewList(ShapeVec);
		InputTensorParams["SubInputShapeE"][0] = he::NewList();
		for(int b=0;b < ThisInput->TensorPointer->shape.size();b++)
		{
			InputTensorParams["SubInputShapeE"][0].append((P["SubInputShapeE"][a][b]-P["SubInputShapeS"][a][b]).i() + P["InputStartShape"][a][b].i());
		}
		DynamicTensor Res = DynamicStdOps_Forward_SubSend({DynamicTensor(CurOps->GradOps)}, InputTensorParams, true);
		BackwardOpsMap[ThisInput.get()][CurOps.get()] = Res.Ops;
	}
}

}