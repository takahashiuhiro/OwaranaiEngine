#include "Linear.h"

/*
*@Params
* DeviceNum 设备计数.
* InFeatures 输入维度.
* OutFeatures 输出维度.
* Bias 是否有偏置.
.*/

void Linear::Init(he InputParams)
{
	Params = InputParams;
	DeviceNum = InputParams["DeviceNum"].i();
	InFeatures = InputParams["InFeatures"].i();
	OutFeatures = InputParams["OutFeatures"].i();
	Bias = InputParams["Bias"].i();
	Weights["Weight"] = DynamicTensor({ InFeatures , OutFeatures }, true, DeviceNum);
	Weights["Weight"].FillRandValUniform(-std::sqrt(1. / InFeatures), std::sqrt(1. / InFeatures));
	if (Bias)
	{
		Weights["Bias"] = DynamicTensor({ 1 , OutFeatures }, true, DeviceNum);
		Weights["Bias"].FillRandValUniform(-std::sqrt(1. / InFeatures), std::sqrt(1. / InFeatures));
	}
}
std::vector<DynamicTensor> Linear::Forward(std::vector<DynamicTensor>InputForwardList, he InputParams)
{
	if (Bias)return { InputForwardList[0] % Weights["Weight"] + Weights["Bias"] };
	return { InputForwardList[0] % Weights["Weight"]};
}