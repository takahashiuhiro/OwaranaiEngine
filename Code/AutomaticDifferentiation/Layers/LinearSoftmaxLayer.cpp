#include "LinearSoftmaxLayer.h"

LinearSoftmaxLayer::LinearSoftmaxLayer(BaseLayer* ParentThis,std::string ThisLayerName,size_t ThisDeviceNum, std::vector<size_t>WeightShape, size_t SoftmaxDim)
{
    this->CommonInit(ParentThis,ThisLayerName,ThisDeviceNum);
    this->RegisterLayer(std::make_shared<LinearLayer>(this, "layer_1", ThisDeviceNum ,WeightShape));
    //this->RegisterLayer(std::make_shared<LinearLayer>(this, "layer_2", ThisDeviceNum ,WeightShape));
}

std::vector<std::string> LinearSoftmaxLayer::Forward(std::vector<std::string>InputNodeArray)
{
    std::vector<std::string> OutputIds_1 = this->SubLayers["layer_1"]->Forward(InputNodeArray);
    //std::vector<std::string> OutputIds_2 = this->SubLayers["layer_2"]->Forward(OutputIds_1);
    return OutputIds_1;
}