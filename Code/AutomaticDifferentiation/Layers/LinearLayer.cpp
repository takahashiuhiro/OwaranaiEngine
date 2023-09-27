#include "LinearLayer.h"

LinearLayer::LinearLayer(BaseLayer* ParentThis,std::string ThisLayerName, std::vector<size_t>WeightShape)
{
    this->CommonInit(ParentThis, ThisLayerName);
}