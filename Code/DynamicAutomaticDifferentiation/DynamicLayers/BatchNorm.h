#pragma once
#include "BaseDynamicLayer.h"

/*
*@Params
* NormalizedShape 归一化.
* Default:
* eps =1e-5 误差.
* ElementwiseAffine = true 元素仿射.
* Bias = true 偏置.
.*/

class BatchNorm : public BaseDynamicLayer
{
public:
	virtual void SetLayerParams();
	virtual void InitContent();
	virtual std::vector<DynamicTensor> Forward(std::vector<DynamicTensor>InputForwardList, he InputParams = he());

    int NumFeatures;
	float eps;
	bool ElementwiseAffine;
	bool Bias;
};