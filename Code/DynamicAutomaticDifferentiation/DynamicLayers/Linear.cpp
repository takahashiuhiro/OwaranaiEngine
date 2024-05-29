#include "Linear.h"

void Linear::SetLayerParams()
{
	InFeatures = Params["InFeatures"].i();
	OutFeatures = Params["OutFeatures"].i();
	if (Params.In("Bias"))Bias = Params["Bias"].i();
	else Bias = true;
}
void Linear::Init(he InputParams)
{
	SetParams(InputParams);
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