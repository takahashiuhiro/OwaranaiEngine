#include "LinearSoftmaxLayer.h"

LinearSoftmaxLayer::LinearSoftmaxLayer(BaseLayer* ParentThis,std::string ThisLayerName, std::vector<size_t>WeightShape, size_t SoftmaxDim)
{
    this->CommonInit(ParentThis);
}