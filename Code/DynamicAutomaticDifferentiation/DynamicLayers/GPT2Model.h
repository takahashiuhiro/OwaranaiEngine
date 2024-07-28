#pragma once
#include "BaseDynamicLayer.h"

class GPT2Model : public BaseDynamicLayer
{
public:
	virtual void SetLayerParams();
	virtual void InitContent();
	virtual std::vector<DynamicTensor> Forward(std::vector<DynamicTensor>InputForwardList, he InputParams = he());

    int BlockSize;
    int VocabSize;
    int NLayers;
    int NHead;
    int NEmbd;
    float Dropout; 
    int Bias;

    std::vector<std::string>TransformerBlockLayerNames;

    static void InitWeights(BaseDynamicLayer* CurLayer);
    static void InitWeightsCProj(BaseDynamicLayer* CurLayer, int ConfigNLayers);

    

};