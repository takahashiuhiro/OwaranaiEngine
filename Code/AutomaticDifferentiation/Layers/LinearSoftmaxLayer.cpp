#include "LinearSoftmaxLayer.h"

LinearSoftmaxLayer::LinearSoftmaxLayer(BaseLayer* ParentThis,std::string ThisLayerName,size_t ThisDeviceNum, std::vector<size_t>WeightShape, size_t SoftmaxDim)
{
    this->CommonInit(ParentThis,ThisLayerName,ThisDeviceNum);
    this->RegisterLayer(std::make_shared<LinearLayer>(this, "layer_1", ThisDeviceNum ,WeightShape));
}