#pragma once
#include "StdInclude.h"
#include "../TensorCore/MoudleInclude.h"
#include "../Ops/MoudleInclude.h"
#include "../ComputationalGraph/MoudleInclude.h"

struct BaseLayer
{
public:

    std::vector<CGNode*>InputCGNode;
    std::vector<CGNode*>ParamsCGNode;

    void LayerInit(std::vector<CGNode*>InputCGNode, std::vector<CGNode*>ParamsCGNode);
};