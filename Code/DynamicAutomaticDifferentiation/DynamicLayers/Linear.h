#pragma once
#include "BaseDynamicLayer.h"

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