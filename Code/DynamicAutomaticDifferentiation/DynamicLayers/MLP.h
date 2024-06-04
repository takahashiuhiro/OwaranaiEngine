#pragma once
#include "BaseDynamicLayer.h"
#include "Linear.h"
#include "LayerNorm.h"

/*
*@Params
* InChannels  输入维度.
* HiddenChannels  隐藏层.
* Default:
* NormLayer = "None" 使用的norm种类.
* ActivationLayer = "ReLU" 激活函数, None不用.
* Bias = true 偏置.
* Dropout = 0 给dropout用的百分比.
.*/

class MLP : public BaseDynamicLayer
{
	virtual void SetLayerParams();
	virtual void InitContent();
	virtual std::vector<DynamicTensor> Forward(std::vector<DynamicTensor>InputForwardList, he InputParams = he());

	int InChannels;
	std::vector<int>HiddenChannels;
	std::string NormLayer;
	std::string ActivationLayer;
	bool Bias;
	float Dropout;

	std::vector<std::string>SubLayerLinearNames;
	std::vector<std::string>SubLayerNormNames;
};