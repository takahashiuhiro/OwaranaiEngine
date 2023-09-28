#include "BaseLayer.h"

bool BaseLayer::IsRootNode()
{
    return ParentLayer == nullptr;
}

void BaseLayer::CommonInit(BaseLayer* InputParentLayer, std::string InputLayerName, size_t ThisDeviceNum)
{
    ParentLayer = InputParentLayer;
    LayerName = InputLayerName;
    DeviceNum = ThisDeviceNum;
    if(IsRootNode())
    {
        CG = std::make_shared<ComputationalGraph>();
    }
    else
    {
        CG = InputParentLayer->CG;
        this->PreName = ParentLayer->PreName + std::string(".") +InputLayerName;
    }
}

void BaseLayer::RegisterLayer(std::shared_ptr<BaseLayer>InputLayer)
{
    SubLayers[InputLayer->LayerName] = InputLayer;
}

void BaseLayer::RegisterWeightNode()
{

}

void BaseLayer::CommonDestroy()
{

}