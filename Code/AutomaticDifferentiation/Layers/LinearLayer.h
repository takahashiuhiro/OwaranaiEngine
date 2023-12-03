#pragma once
#include "BaseLayer.h"

class LinearLayer:public BaseLayer
{
public:
    LinearLayer(){};
    LinearLayer(BaseLayer* ParentThis,std::string ThisLayerName, size_t ThisDeviceNum, size_t InFeatures,size_t OutFeatures, bool Bias=true);

    virtual std::vector<std::string> Forward(std::vector<std::string>InputNodeArray);

    size_t InFeatures;
    size_t OutFeatures;
    bool Bias;
};