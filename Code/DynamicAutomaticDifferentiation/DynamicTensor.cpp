#include "DynamicTensor.h"
#include "../CommonMathMoudle/OpsType.h"

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
	if(Ops->LeafNode == this)Ops->LeafNode = nullptr;
}

DynamicTensor DynamicTensor::GetGrad()
{
	Log::Assert(Ops->GradOps != nullptr, "This DynamicTensor Has No Grad");
	return DynamicTensor(Ops->GradOps);
}

void DynamicTensor::PrintData()
{
	Ops->TensorPointer->PrintData();
}

void DynamicTensor::Fill(float InputValue)
{
	Ops->TensorPointer->FillArray(InputValue);
}

DynamicTensor DynamicTensor::SetComputationalHistory(Tensor* ResTensor, std::vector<DynamicTensor>InputList, he InputPrams, size_t InputOpsType, bool RequiresGrad)
{
	bool MaxRequiresGrad = 0,MaxIsEval = 0;
	for (size_t a = 0; a < InputList.size(); a++)MaxRequiresGrad |= InputList[a].Ops->RequiresGrad;
	for (size_t a = 0; a < InputList.size(); a++)MaxIsEval |= InputList[a].Ops->IsEval;
	DynamicTensor Res(std::shared_ptr<Tensor>(ResTensor), MaxRequiresGrad&RequiresGrad&(!MaxIsEval));
	if (!RequiresGrad)return Res;
	Res.Ops->DynamicOpsType = InputOpsType;
	Res.Ops->Params = InputPrams;
	for (size_t a = 0; a < InputList.size(); a++)
	{
		Res.Ops->InputOpsList.push_back(InputList[a].Ops);
		InputList[a].Ops->OutputOpsSet.insert(Res.Ops.get());
	}
	return Res;
}

void DynamicTensor::Backward(DynamicTensor* Loss)
{
	Log::Assert(Ops->OutputOpsSet.size() == 0, "DynamicTensor Backward Must Be Output Data");
	std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>BackwardOpsMap;
	std::map<DynamicOps*, std::set<DynamicOps*>>OutputSetSize;
	GetAllOutputSizeBeforeBackward(OutputSetSize, Ops);
	BackwardDFS(BackwardOpsMap, OutputSetSize,Loss, Ops);
}

void DynamicTensor::BackwardDFS(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::map<DynamicOps*, std::set<DynamicOps*>>& OutputSetSize, DynamicTensor* Loss, std::shared_ptr<DynamicOps>CurOps)
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
				ThisOpsGradRes = DynamicTensor::DynamicStdOps_Forward_Add({ ThisOpsGradRes, DynamicTensor(BackwardOpsMap[CurOps.get()][OutputList[a]]) },he(), true);
			}
			auto DynamicTensorGrad = std::make_shared<DynamicTensor>(ThisOpsGradRes.Ops);
			CurOps->GradOps = DynamicTensorGrad->Ops;
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

void DynamicTensor::GenEmptyGradDynamicTensor(DynamicTensor* Loss)
{
	Tensor* GradResTensor;
	if(Loss == nullptr)GradResTensor = Ops->TensorPointer->Copy();
	else GradResTensor = Loss->Ops->TensorPointer->Copy();
	auto DynamicTensorGrad = std::make_shared<DynamicTensor>(std::shared_ptr<Tensor>(GradResTensor), Ops->RequiresGrad);
	Grad = DynamicTensorGrad;
	Ops->GradOps = Grad->Ops;
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

DynamicTensor DynamicTensor::operator+(DynamicTensor Other)
{
	if (Other.Ops.get() != Ops.get())
	{
		Log::Assert(Ops->TensorPointer->shape.size() == Other.Ops->TensorPointer->shape.size(), "DynamicTensor Opeator+ Shape Error,Shape Num!=");
		he BCParams = he::NewDict();
		BCParams["BroadCastToShape"] = he::NewList();
		int BCFlag = 0;
		for (size_t a = 0; a < Ops->TensorPointer->shape.size(); a++)
		{
			int ThisShapeNum = Ops->TensorPointer->shape[a];
			int OtherShapeNum = Other.Ops->TensorPointer->shape[a];
			Log::Assert(!(ThisShapeNum != OtherShapeNum &&std::min(ThisShapeNum, OtherShapeNum)!=1), "DynamicTensor Opeator+ Shape Error, Dim Can Not BroadCast");
			BCParams["BroadCastToShape"].append(std::max(ThisShapeNum, OtherShapeNum));
			if (ThisShapeNum < OtherShapeNum)BCFlag |= 1;
			if (ThisShapeNum > OtherShapeNum)BCFlag |= 2;
		}
		if(!BCFlag)return DynamicStdOps_Forward_Add({ *this, Other }, he(), true);
		if (BCFlag == 1)return DynamicStdOps_Forward_Add({ DynamicStdOps_Forward_BroadCastTo({*this}, BCParams,true), Other}, he(), true);
		if (BCFlag == 2)return DynamicStdOps_Forward_Add({ *this, DynamicStdOps_Forward_BroadCastTo({Other}, BCParams,true) }, he(), true);
		return DynamicStdOps_Forward_Add({ DynamicStdOps_Forward_BroadCastTo({*this}, BCParams,true), DynamicStdOps_Forward_BroadCastTo({Other},BCParams,true) }, he(), true);
	}
	else
	{
		return DynamicStdOps_Forward_Add({ *this, DynamicStdOps_Forward_Add({Other},he(),true) }, he(), true);
	}
}

DynamicTensor DynamicTensor::operator%(DynamicTensor Other)
{
	if (Other.Ops.get() != Ops.get())
	{
		//todo::检测广播
		he MatmulParams = he::NewDict();
		MatmulParams["is_input_1st_T"] = false;
		MatmulParams["is_input_2nd_T"] = false;
		return DynamicStdOps_Forward_Matmul({ *this, Other }, MatmulParams, true);
	}
	else
	{
		he MatmulParams = he::NewDict();
		MatmulParams["is_input_1st_T"] = false;
		MatmulParams["is_input_2nd_T"] = false;
		return DynamicStdOps_Forward_Matmul({ *this, DynamicStdOps_Forward_Add({Other},he(),true) }, MatmulParams, true);
	}
}
