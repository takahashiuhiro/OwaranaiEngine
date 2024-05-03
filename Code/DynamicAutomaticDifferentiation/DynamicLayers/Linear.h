#pragma once
#include "BaseDynamicLayer.h"

class Linear :public BaseDynamicLayer
{
public:
	virtual void Init(he InputParams);
	virtual std::vector<DynamicTensor> Forward(std::vector<DynamicTensor>InputForwardList, he InputParams = he());

	size_t InFeatures;
	size_t OutFeatures;
	bool Bias;
};