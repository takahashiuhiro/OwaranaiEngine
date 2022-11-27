#include "BaseLayer.h"

void BaseLayer::LayerInit(std::vector<CGNode*>InputCGNode)
{
    this->InputCGNode = InputCGNode;
}

void BaseLayer::Freeze()
{
    for(int a =0;a<ParamsCGNode.size();a++)
    {
        ParamsCGNode[a]->NodeType["Freeze"] = 1;
    }
}