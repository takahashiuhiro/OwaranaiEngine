#include "BaseDynamicLayer.h"

std::vector<DynamicTensor> BaseDynamicLayer::Parameters()
{
	std::vector<DynamicTensor> Res = {};
	for (auto& it : Weights)Res.push_back(it.second);
	for (auto& it : SubLayers)
	{
		auto TMPRes = it.second->Parameters();
		for (size_t a = 0; a < TMPRes.size(); a++)Res.push_back(TMPRes[a]);
	}
	return Res;
}

std::map<std::string, DynamicTensor> BaseDynamicLayer::StateDict()
{
	std::map<std::string, DynamicTensor>ResMp = {};
	std::string PreStr = "";
	StateDictDFS(ResMp, PreStr);
	return ResMp;
}

void BaseDynamicLayer::StateDictDFS(std::map<std::string, DynamicTensor>& ResMp, std::string PreStr)
{
	for (auto& it : Weights)
	{
		ResMp[PreStr + it.first] = it.second;
	}
	for (auto& it : SubLayers)
	{
		it.second->StateDictDFS(ResMp, PreStr + it.first + std::string("."));
	}
}

void BaseDynamicLayer::Eval()
{
	IsEval = true;
	auto WeightsVec = Parameters();
	for (auto& it : WeightsVec)it.Ops->IsEval = IsEval;
}
void BaseDynamicLayer::Train()
{
	IsEval = false;
	auto WeightsVec = Parameters();
	for (auto& it : WeightsVec)it.Ops->IsEval = IsEval;
}

void BaseDynamicLayer::SetCommonDefaultParams()
{
	if (Params.In("DeviceNum"))DeviceNum = Params["DeviceNum"].i();
	else DeviceNum = 0;
}

void BaseDynamicLayer::SetParams(he InputParams)
{
	Params = InputParams;
	SetCommonDefaultParams();
	SetLayerParams();
}

void BaseDynamicLayer::Init(he InputParams)
{
	SetParams(InputParams);
	InitContent();
}

void BaseDynamicLayer::Save(std::string InputName)
{
	auto StateDictParamsMp = StateDict();
	std::ofstream OpenedFile(InputName, std::ios::binary);
	Log::Assert(OpenedFile.is_open(), std::string("This File Is Not Opened::") + InputName);
	size_t StateDictSize = StateDictParamsMp.size();
	OpenedFile.write(reinterpret_cast<const char*>(&StateDictSize), sizeof(StateDictSize));
	for(auto&it:StateDictParamsMp)
	{
		SaveToFileString(OpenedFile, it.first);
		it.second.Ops->TensorPointer->SaveToFile(OpenedFile);
	}
	OpenedFile.close();
}

void BaseDynamicLayer::Load(std::string InputName)
{
	auto StateDictParamsMp = StateDict();
	std::ifstream OpenedFile(InputName, std::ios::binary);
	Log::Assert(OpenedFile.is_open(), std::string("This File Is Not Opened::") + InputName);
	size_t StateDictSize;
	OpenedFile.read(reinterpret_cast<char*>(&StateDictSize), sizeof(StateDictSize));
	for(size_t a = 0;a<StateDictSize;a++)
	{
		std::string WeightName = LoadFromFileString(OpenedFile);
		StateDictParamsMp[WeightName].Ops->TensorPointer->LoadFromFile(OpenedFile);
	}
}

float BaseDynamicLayer::GetNumParams()
{
	float Res = 0;
	for(auto&it:Parameters())Res += it.Numel()*1e-6;
	return Res;
}