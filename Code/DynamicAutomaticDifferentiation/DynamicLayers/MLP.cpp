#include "MLP.h"

void MLP::SetLayerParams()
{
	InChannels = Params["InChannels"].i();
	Params["HiddenChannels"].v(HiddenChannels);
	if (Params.In("NormLayer"))NormLayer = Params["NormLayer"].s();
	else NormLayer = "None";
	if (Params.In("ActivationLayer"))ActivationLayer = Params["ActivationLayer"].s();
	else ActivationLayer = "ReLU";
	if (Params.In("Bias"))Bias = Params["Bias"].i();
	else Bias = true;
	if (Params.In("Dropout"))Dropout = Params["Dropout"].f();
	else Dropout = 0;
}
void MLP::InitContent()
{
	for (size_t a = 0; a < HiddenChannels.size(); a++)
	{
		he LinearParam = he::NewDict();
		if (!a)LinearParam["InFeatures"] = InChannels;
		else LinearParam["InFeatures"] = HiddenChannels[a - 1];
		LinearParam["OutFeatures"] = HiddenChannels[a];
		LinearParam["Bias"] = Bias;
		std::string ThisLinearName = std::string("Linear") + NumberToString(a);
		CreateNewLayer<Linear>(ThisLinearName, LinearParam);
		SubLayerLinearNames.push_back(ThisLinearName);
		if (NormLayer != "None")
		{
			he NormParam = he::NewDict();
			std::string ThisNormName = NormLayer + NumberToString(a);
			if (NormLayer == "LayerNorm")
			{
				NormParam["NormalizedShape"] = he::NewList<int>({ HiddenChannels[a] });
				CreateNewLayer<LayerNorm>(ThisNormName, NormParam);
			}
			SubLayerNormNames.push_back(ThisNormName);
		}
	}
}
std::vector<DynamicTensor> MLP::Forward(std::vector<DynamicTensor>InputForwardList, he InputParams)
{
	auto x = InputForwardList[0];
	for (size_t a = 0; a < SubLayerLinearNames.size(); a++)
	{
		x = SubLayers[SubLayerLinearNames[a]]->Forward({ x })[0];
		if(NormLayer != "None")x = SubLayers[SubLayerNormNames[a]]->Forward({ x })[0];
		if (ActivationLayer == "GELU")x = x.GELU();
		x = DynamicTensor::Dropout(x, Dropout);
	}
	return { x };
}