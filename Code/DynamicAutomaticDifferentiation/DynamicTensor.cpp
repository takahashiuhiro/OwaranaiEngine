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
}

void DynamicTensor::OpsSetInMap()
{
	BackwardOps[OpsType::Add] = DynamicStdOps_Backward_Add;
}

DynamicTensor::~DynamicTensor()
{
	if(Ops->LeafNode == this)Ops->LeafNode = nullptr;
}

DynamicTensor DynamicTensor::SetComputationalHistory(Tensor* ResTensor, std::vector<DynamicTensor>InputList, he InputPrams, size_t InputOpsType, bool RequiresGrad)
{
	bool MaxRequiresGrad = 0;
	for (size_t a = 0; a < InputList.size(); a++)MaxRequiresGrad |= InputList[a].Ops->RequiresGrad;
	DynamicTensor Res(std::shared_ptr<Tensor>(ResTensor), MaxRequiresGrad&RequiresGrad);
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
	std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>BackwardOpsMap;
	std::map<DynamicOps*, int>OutputSetSize;
	GetAllOutputSizeBeforeBackward(OutputSetSize, Ops);
	BackwardDFS(BackwardOpsMap, OutputSetSize,Loss, Ops);
}

void DynamicTensor::BackwardDFS(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::map<DynamicOps*, int>& OutputSetSize, DynamicTensor* Loss, std::shared_ptr<DynamicOps>CurOps)
{
	if (CheckPartialGradReady(BackwardOpsMap,OutputSetSize, CurOps))
	{
		if (CurOps->OutputOpsSet.empty())
		{
			GenEmptyGradDynamicTensor(Loss);
		}
		else
		{
			std::vector<DynamicOps*>OutputList;
			for (auto Item : CurOps->OutputOpsSet)OutputList.push_back(Item);
			DynamicTensor ThisOpsGradRes = DynamicTensor(BackwardOpsMap[CurOps.get()][OutputList[0]]);
			for (size_t a = 1; a < OutputList.size(); a++)
			{
				ThisOpsGradRes = DynamicTensor::DynamicStdOps_Forward_Add({ ThisOpsGradRes, DynamicTensor(BackwardOpsMap[CurOps.get()][OutputList[a]]) },he(), true);
			}
			auto DynamicTensorGrad = std::make_shared<DynamicTensor>(ThisOpsGradRes.Ops);
			Grad = DynamicTensorGrad;
			CurOps->GradOps = Grad->Ops;
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

bool DynamicTensor::CheckPartialGradReady(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>&BackwardOpsMap, std::map<DynamicOps*, int>& OutputSetSize, std::shared_ptr<DynamicOps>CurOps)
{
	if (BackwardOpsMap.find(CurOps.get()) == BackwardOpsMap.end())
	{
		BackwardOpsMap[CurOps.get()] = {};
	}
	return BackwardOpsMap[CurOps.get()].size() == OutputSetSize[CurOps.get()];
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

void DynamicTensor::GetAllOutputSizeBeforeBackward(std::map<DynamicOps*, int>& OutputSetSize, std::shared_ptr<DynamicOps>CurOps)
{
	if (OutputSetSize.find(CurOps.get()) != OutputSetSize.end())return;
	OutputSetSize[CurOps.get()] = CurOps->OutputOpsSet.size();
	for (size_t a = 0; a < CurOps->InputOpsList.size(); a++)
	{
		GetAllOutputSizeBeforeBackward(OutputSetSize, CurOps->InputOpsList[a]);
	}
}

DynamicTensor DynamicTensor::DynamicStdOps_Forward_Add(std::vector<DynamicTensor>InputList, he InputPrams, bool RequiresGrad)
{
	auto ResTensorContent = InputList[0].Ops->TensorPointer->Copy();
	for (size_t a = 1; a < InputList.size(); a++)
	{
		Tensor* TMPTensor = ResTensorContent->Add(InputList[a].Ops->TensorPointer.get());
		delete ResTensorContent;
		ResTensorContent = TMPTensor;
	}
	return SetComputationalHistory(ResTensorContent, InputList, InputPrams,OpsType::Add, RequiresGrad);
}
void DynamicTensor::DynamicStdOps_Backward_Add(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::shared_ptr<DynamicOps>CurOps)
{
	for (size_t a = 0; a < CurOps->InputOpsList.size(); a++)
	{
		if (!CurOps->InputOpsList[a]->RequiresGrad)continue;
		if(BackwardOpsMap.find(CurOps->InputOpsList[a].get()) == BackwardOpsMap.end())
		{
			BackwardOpsMap[CurOps->InputOpsList[a].get()] = {};
		}
		auto AddRes = DynamicTensor::DynamicStdOps_Forward_Add({DynamicTensor(CurOps->GradOps)},he(),true);
		BackwardOpsMap[CurOps->InputOpsList[a].get()][CurOps.get()] =AddRes.Ops;
	}
}
