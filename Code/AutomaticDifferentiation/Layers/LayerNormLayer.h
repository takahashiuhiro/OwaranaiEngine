#pragma once
#include "BaseLayer.h"

class LayerNormLayer:public BaseLayer
{
public:
    LayerNormLayer(){};
    LayerNormLayer(BaseLayer* ParentThis,std::string ThisLayerName, size_t ThisDeviceNum,std::vector<size_t>WeightShape,size_t UseNum, bool ElementwiseAffine = true, bool HasBias = true, float eps = 1e-5);

    bool ElementwiseAffine;
    bool HasBias;
    std::vector<size_t>WeightShape;
    size_t UseNum;
    float eps;

    virtual std::vector<std::string> Forward(std::vector<std::string>InputNodeArray);
};