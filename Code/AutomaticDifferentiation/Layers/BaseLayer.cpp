#include "BaseLayer.h"

bool BaseLayer::IsRootNode()
{
    return ParentLayer == nullptr;
}

void BaseLayer::CommonInit(BaseLayer* InputParentLayer)
{
    ParentLayer = InputParentLayer;
    if(IsRootNode())
    {
        CG = std::make_shared<ComputationalGraph>();
    }
    else
    {
        CG = InputParentLayer->CG;
    }
}