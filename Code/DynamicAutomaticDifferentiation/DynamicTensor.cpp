#include "DynamicTensor.h"


DynamicTensor::DynamicTensor(){};
DynamicTensor::DynamicTensor(std::shared_ptr<Tensor> InputTensorPointer, int InputRequiresGrad)
{
	RequiresGrad = InputRequiresGrad;
	TensorPointer = InputTensorPointer;
	Ops = std::make_shared<DynamicOps>();//每个动态张量都需要算子
	Ops->LeafNode = this;//算子连回自家张量的途径
	if (InputTensorPointer)
	{
		auto GradTensor = std::shared_ptr<Tensor>(InputTensorPointer->CopyNewEmptyTensor());
		GradTensor->FillArray(0);
		Grad = std::make_shared<DynamicTensor>(GradTensor, InputRequiresGrad-1);
		Ops->GradOps = Grad->Ops;//算子直接找到导数算子
	}
}

void DynamicTensor::OpsSetInMap()
{
	ForwardOpsMap[OpsType::Add] = Add;
}

DynamicTensor::~DynamicTensor()
{
	Ops->LeafNode = nullptr;//此时算子和动态张量会脱离
}

DynamicTensor DynamicTensor::SetComputationalHistory(Tensor* ResTensor, std::vector<DynamicTensor*>InputList, he InputPrams, size_t InputOpsType, bool RequiresGrad)
{
	if (!RequiresGrad)return DynamicTensor(std::shared_ptr<Tensor>(ResTensor));
	int MaxRequiresGrad = 0;
	for (size_t a = 0; a < InputList.size(); a++)MaxRequiresGrad = std::max(MaxRequiresGrad, InputList[a]->RequiresGrad);
	DynamicTensor Res = DynamicTensor(std::shared_ptr<Tensor>(ResTensor), MaxRequiresGrad);
	Res.Ops->DynamicOpsType = InputOpsType;
	Res.Ops->Params = InputPrams;
	for (size_t a = 0; a < InputList.size(); a++)
	{
		Res.Ops->InputOpsList.push_back(InputList[a]->Ops);
		InputList[a]->Ops->OutputOpsSet.insert(Res.Ops.get());
	}
	return Res;
}

DynamicTensor DynamicTensor::Add(std::vector<DynamicTensor*>InputList, he InputPrams, bool RequiresGrad)
{
	auto ResTensorContent = InputList[0]->TensorPointer->Add(InputList[1]->TensorPointer.get());
	return SetComputationalHistory(ResTensorContent, InputList, InputPrams,OpsType::Add, RequiresGrad);
}