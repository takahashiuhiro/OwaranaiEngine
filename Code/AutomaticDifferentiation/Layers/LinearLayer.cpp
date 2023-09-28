#include "LinearLayer.h"

LinearLayer::LinearLayer(BaseLayer* ParentThis,std::string ThisLayerName, size_t ThisDeviceNum, std::vector<size_t>WeightShape)
{
    this->CommonInit(ParentThis, ThisLayerName, ThisDeviceNum);
}