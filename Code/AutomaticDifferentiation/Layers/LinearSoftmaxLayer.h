#pragma once
#include "BaseLayer.h"
#include "LinearLayer.h"

class LinearSoftmaxLayer:public BaseLayer
{
public:
    LinearSoftmaxLayer(){};
    LinearSoftmaxLayer(BaseLayer* ParentThis,std::string ThisLayerName, std::vector<size_t>WeightShape, size_t SoftmaxDim);
};