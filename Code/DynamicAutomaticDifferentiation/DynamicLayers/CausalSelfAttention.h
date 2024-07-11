#pragma once
#include "BaseDynamicLayer.h"

/*
*@Params
* NHead 多头.
* NEmbd 嵌入长度.
* Dropout dropout比例.
* BlockSize 下三角阵的大小.
* Bias 偏置.
.*/

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