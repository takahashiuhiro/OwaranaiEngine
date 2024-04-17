#include "DynamicTensor.h"


DynamicTensor::DynamicTensor(){};
DynamicTensor::DynamicTensor(std::shared_ptr<Tensor> InputTensorPointer, bool InputRequiresGrad)
{
	RequiresGrad = InputRequiresGrad;
	Ops = std::make_shared<DynamicOps>();//每个动态张量都需要算子
	Ops->TensorPointer = InputTensorPointer;
	Ops->LeafNode = this;//算子连回自家张量的途径
}

void DynamicTensor::OpsSetInMap()
{
	ForwardOpsMap[OpsType::Add] = Add;
}

DynamicTensor::~DynamicTensor()
{
	Ops->LeafNode = nullptr;//此时算子和动态张量会脱离
}

DynamicTensor DynamicTensor::SetComputationalHistory(Tensor* ResTensor, std::vector<DynamicTensor>InputList, he InputPrams, size_t InputOpsType, bool RequiresGrad)
{
	bool MaxRequiresGrad = 0;
	for (size_t a = 0; a < InputList.size(); a++)MaxRequiresGrad |= InputList[a].RequiresGrad;
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

void DynamicTensor::Backward()
{

}



DynamicTensor DynamicTensor::Add(std::vector<DynamicTensor>InputList, he InputPrams, bool RequiresGrad)
{
	auto ResTensorContent = InputList[0].Ops->TensorPointer->Add(InputList[1].Ops->TensorPointer.get());
	return SetComputationalHistory(ResTensorContent, InputList, InputPrams,OpsType::Add, RequiresGrad);
}

