#pragma once
#include "BaseDynamicLayer.h"

namespace OwaranaiEngine
{

/*
*@Params
* NormalizedShape 归一化.
* Default:
* eps =1e-5 误差.
* ElementwiseAffine = true 元素仿射.
* Bias = true 偏置.
.*/

class LayerNorm : public BaseDynamicLayer
{
public:
	virtual void SetLayerParams();
	virtual void InitContent();
	virtual std::vector<DynamicTensor> Forward(std::vector<DynamicTensor>InputForwardList, he InputParams = he());

	std::vector<size_t>NormalizedShape;
	float eps;
	bool ElementwiseAffine;
	bool Bias;
};

void LayerNorm::SetLayerParams()
{
	Params["NormalizedShape"].v(NormalizedShape);
	if (Params.In("eps"))eps = Params["eps"].f();
	else eps = 1e-5;
	if (Params.In("ElementwiseAffine"))ElementwiseAffine = Params["ElementwiseAffine"].i();
	else ElementwiseAffine = true;
	if (Params.In("Bias"))Bias = Params["Bias"].i();
	else Bias = true;
}
void LayerNorm::InitContent()
{
	if (ElementwiseAffine)
	{
		Weights["Weight"] = DynamicTensor(NormalizedShape, true, DeviceNum);
		Weights["Weight"].Fill(1);
		if (Bias)
		{
			Weights["Bias"] = DynamicTensor(NormalizedShape, true, DeviceNum);
			Weights["Bias"].Fill(0);
		}
	}
}
std::vector<DynamicTensor> LayerNorm::Forward(std::vector<DynamicTensor>InputForwardList, he InputParams)
{
	std::vector<int>MeanDims;
	for (size_t a = InputForwardList[0].Shape().size() - NormalizedShape.size(); a < InputForwardList[0].Shape().size(); a++)MeanDims.push_back(a);
	auto Res = (InputForwardList[0] - InputForwardList[0].Mean(MeanDims, true))*(InputForwardList[0].Var(MeanDims, true,0) + eps).Pow(-0.5);
	if (!ElementwiseAffine)return { Res };
	if (!Bias)return { Res * Weights["Weight"] };
	return { Res * Weights["Weight"] + Weights["Bias"] };
}

}