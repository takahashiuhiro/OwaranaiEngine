#pragma once
#include "BaseLayer.h"

class LinearLayer:public BaseLayer
{
public:
    LinearLayer(){};
    LinearLayer(BaseLayer* ParentThis,std::string ThisLayerName, size_t ThisDeviceNum,std::vector<size_t>WeightShape);

    virtual std::vector<std::string> Forward(std::vector<std::string>InputNodeArray);
};