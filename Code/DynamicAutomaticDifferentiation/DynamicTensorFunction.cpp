#include "DynamicTensor.h"

DynamicTensor DynamicTensor::Sum(std::vector<int>Dims, bool KeepDim)
{
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