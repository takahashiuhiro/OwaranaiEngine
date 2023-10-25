#pragma once
#include "BaseLayer.h"

class LayerNormLayer:public BaseLayer
{
public:
    LayerNormLayer(){};
    LayerNormLayer(BaseLayer* ParentThis,std::string ThisLayerName, size_t ThisDeviceNum,std::vector<size_t>WeightShape,size_t UseNum, bool ElementwiseAffine = true);

    virtual std::vector<std::string> Forward(std::vector<std::string>InputNodeArray);
};