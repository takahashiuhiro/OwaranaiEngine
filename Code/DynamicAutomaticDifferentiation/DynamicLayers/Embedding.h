#pragma once
#include "BaseDynamicLayer.h"

class Embedding : public BaseDynamicLayer
{
public:
	virtual void SetLayerParams();
	virtual void InitContent();
	virtual std::vector<DynamicTensor> Forward(std::vector<DynamicTensor>InputForwardList, he InputParams = he());

	int NumEmbeddings;
    int EmbeddingDim;
    int PaddingIdx;
    float MaxNorm;
    float NormType;
    int ScaleGradByFreq; 
    int Sparse;

    //是否设置了该参数
    bool PaddingIdxSwitch;
    bool MaxNormSwitch;
};