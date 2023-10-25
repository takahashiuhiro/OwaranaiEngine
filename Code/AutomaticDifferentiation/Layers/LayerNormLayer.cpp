#include "LayerNormLayer.h"

LayerNormLayer::LayerNormLayer(BaseLayer* ParentThis,std::string ThisLayerName, size_t ThisDeviceNum,std::vector<size_t>WeightShape,size_t UseNum, bool ElementwiseAffine)
{
    this->CommonInit(ParentThis,ThisLayerName,ThisDeviceNum);
    std::cout<<ElementwiseAffine<<std::endl;
}

std::vector<std::string> LayerNormLayer::Forward(std::vector<std::string>InputNodeArray)
{
    
}