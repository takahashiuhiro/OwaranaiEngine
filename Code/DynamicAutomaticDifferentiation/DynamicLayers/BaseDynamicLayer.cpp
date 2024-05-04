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