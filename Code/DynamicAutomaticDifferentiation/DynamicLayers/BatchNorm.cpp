#include "BatchNorm.h"

void BatchNorm::SetLayerParams()
{
    NumFeatures = Params["NumFeatures"].i();
	if (Params.In("eps"))eps = Params["eps"].f();
	else eps = 1e-5;
	if (Params.In("ElementwiseAffine"))ElementwiseAffine = Params["ElementwiseAffine"].i();
	else ElementwiseAffine = true;
	if (Params.In("Bias"))Bias = Params["Bias"].i();
	else Bias = true;
}
void BatchNorm::InitContent()
{
	if (ElementwiseAffine)
	{
		Weights["Weight"] = DynamicTensor({1,(size_t)NumFeatures,1 }, true, DeviceNum);
		Weights["Weight"].Fill(1);
		if (Bias)
		{
			Weights["Bias"] = DynamicTensor({1,(size_t)NumFeatures,1 }, true, DeviceNum);
			Weights["Bias"].Fill(0);
		}
	}
}
std::vector<DynamicTensor> BatchNorm::Forward(std::vector<DynamicTensor>InputForwardList, he InputParams)
{
	std::vector<int>MeanDims = {0,2};
    std::vector<int>ProtoShape;
    for(auto&it:InputForwardList[0].Shape())ProtoShape.push_back(it);
    auto ViewX = InputForwardList[0].View({ProtoShape[0],ProtoShape[1], -1});
	auto Res = (ViewX - ViewX.Mean(MeanDims, true))*(ViewX.Var(MeanDims, true,0) + eps).Pow(-0.5);
	if (!ElementwiseAffine)return { Res.View(ProtoShape) };
	if (!Bias)return { Res.View(ProtoShape) * Weights["Weight"] };
	return { Res.View(ProtoShape) * Weights["Weight"] + Weights["Bias"] };
}