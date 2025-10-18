#pragma once
#include "BaseDynamicLayer.h"

namespace OwaranaiEngine
{


/*
*@Params
* InFeatures 输入维度.
* OutFeatures 输出维度.
* Default:
* Bias = true 是否有偏置.
.*/

class Linear :public BaseDynamicLayer
{
public:
	virtual void SetLayerParams();
	virtual void InitContent();
	virtual std::vector<DynamicTensor> Forward(std::vector<DynamicTensor>InputForwardList, he InputParams = he());

	size_t InFeatures;
	size_t OutFeatures;
	bool Bias;
};

void Linear::SetLayerParams()
{
	InFeatures = Params["InFeatures"].i();
	OutFeatures = Params["OutFeatures"].i();
	if (Params.In("Bias"))Bias = Params["Bias"].i();
	else Bias = true;
}
void Linear::InitContent()
{
	Weights["Weight"] = DynamicTensor({ OutFeatures,InFeatures }, true, DeviceNum);
	Weights["Weight"].FillRandValUniform(-std::sqrt(1. / InFeatures), std::sqrt(1. / InFeatures));
	if (Bias)
	{
		Weights["Bias"] = DynamicTensor({ 1 , OutFeatures }, true, DeviceNum);
		Weights["Bias"].FillRandValUniform(-std::sqrt(1. / InFeatures), std::sqrt(1. / InFeatures));
	}
}
std::vector<DynamicTensor> Linear::Forward(std::vector<DynamicTensor>InputForwardList, he InputParams)
{
	if (Bias)return { InputForwardList[0] % Weights["Weight"].Transpose(-1,-2) + Weights["Bias"] };
	return { InputForwardList[0] % Weights["Weight"].Transpose(-1,-2)};
}

}