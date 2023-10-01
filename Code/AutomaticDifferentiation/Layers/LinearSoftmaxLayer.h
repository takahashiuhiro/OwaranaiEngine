#pragma once
#include "BaseLayer.h"
#include "LinearLayer.h"

class LinearSoftmaxLayer:public BaseLayer
{
public:
    LinearSoftmaxLayer(){};
    LinearSoftmaxLayer(BaseLayer* ParentThis,std::string ThisLayerName,size_t ThisDeviceNum, std::vector<size_t>WeightShape, size_t SoftmaxDim);

    virtual std::vector<std::string> Forward(std::vector<std::string>InputNodeArray);
};