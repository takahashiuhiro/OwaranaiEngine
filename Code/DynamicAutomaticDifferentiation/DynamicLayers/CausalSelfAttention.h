#pragma once
#include "BaseDynamicLayer.h"

class CausalSelfAttention : public BaseDynamicLayer
{
    virtual void SetLayerParams();
	virtual void InitContent();
	virtual std::vector<DynamicTensor> Forward(std::vector<DynamicTensor>InputForwardList, he InputParams = he());
 
	int NHead;
	int NEmbd;
	float Dropout;
	int BlockSize;
	int Bias;
    
};