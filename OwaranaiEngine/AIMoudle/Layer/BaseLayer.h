#pragma once
#include "StdInclude.h"
#include "../TensorCore/MoudleInclude.h"
#include "../Ops/MoudleInclude.h"
#include "../ComputationalGraph/MoudleInclude.h"

struct BaseLayer
{
public:

    /**input list of layer*/
    std::vector<CGNode*>InputCGNode;
    /**param matrix list*/
    std::vector<CGNode*>ParamsCGNode;
    /**forward output*/
    CGNode* ForwardNode;
    /**init by InputCGNode list*/
    void LayerInit(std::vector<CGNode*>InputCGNode);

    void Freeze();
};