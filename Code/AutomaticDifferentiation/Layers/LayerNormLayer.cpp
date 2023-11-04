#include "LayerNormLayer.h"

LayerNormLayer::LayerNormLayer(BaseLayer* ParentThis,std::string ThisLayerName, size_t ThisDeviceNum,std::vector<size_t>WeightShape,size_t UseNum, bool ElementwiseAffine)
{
    this->CommonInit(ParentThis,ThisLayerName,ThisDeviceNum);
    this->ElementwiseAffine = ElementwiseAffine;
    this->WeightShape = WeightShape;
    this->UseNum = UseNum;
    if(ElementwiseAffine)
    {
        this->RegisterWeightNode("Weight", WeightShape);
        this->RegisterWeightNode("Bias", WeightShape);
    }
}

std::vector<std::string> LayerNormLayer::Forward(std::vector<std::string>InputNodeArray)
{
    //todo
    return {};
}