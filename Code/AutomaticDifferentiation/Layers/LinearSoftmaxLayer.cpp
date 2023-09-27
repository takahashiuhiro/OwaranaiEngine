#include "LinearSoftmaxLayer.h"

LinearSoftmaxLayer::LinearSoftmaxLayer(BaseLayer* ParentThis,std::string ThisLayerName, std::vector<size_t>WeightShape, size_t SoftmaxDim)
{
    this->CommonInit(ParentThis,ThisLayerName);
    this->RegisterLayer(std::make_shared<LinearLayer>(this, "layer_1", WeightShape));
}