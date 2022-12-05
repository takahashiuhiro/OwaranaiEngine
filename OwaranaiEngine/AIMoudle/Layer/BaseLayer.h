#pragma once
#include "StdInclude.h"
#include "Hyperparameter.h"
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
    /**Hyperparameter*/
    Hyperparameter Params;
    /**forward output*/
    CGNode* ForwardNode;
    /**init by InputCGNode list*/
    void LayerInit(std::vector<CGNode*>InputCGNode);
    /**get the result of the episode*/
    CGNode* Forward();
    void Backward(Tensor* Loss);
    /**not to update this layer's params matrix*/
    void Freeze();
    /**layer class must override this function*/
    virtual void ForwardBuild() = 0;
};