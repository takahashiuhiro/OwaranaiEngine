#pragma once
#include "BaseLayer.h"

class LinearLayer:public BaseLayer
{
public:
    LinearLayer(){};
    LinearLayer(BaseLayer* ParentThis,std::string ThisLayerName, std::vector<size_t>WeightShape);
};