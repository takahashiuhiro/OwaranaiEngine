#include "DynamicTensor.h"

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
	for (size_t a = 0; a < SplitSize; a++)
	{
		if (a < (SplitSize - ProtoDimSize % SplitSize))SplitSections.push_back(ProtoDimSize / SplitSize);
		else SplitSections.push_back(ProtoDimSize / SplitSize + 1);
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

//DynamicTensor DynamicTensor::Tanh()
//{
//
//}