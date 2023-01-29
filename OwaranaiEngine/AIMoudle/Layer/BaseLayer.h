#pragma once
#include "StdInclude.h"
#include "../Helpers/MoudleInclude.h"
#include "../TensorCore/MoudleInclude.h"
#include "../Ops/MoudleInclude.h"
#include "../ComputationalGraph/MoudleInclude.h"
#include "../LossFunction/MoudleInclude.h"
#include "../Optimizer/MoudleInclude.h"

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
    BaseLoss* LossFunctionPointer;
    void Backward(Tensor* Loss);
    /**not to update this layer's params matrix*/
    void Freeze();
    /**layer class must override this function*/
    virtual void ForwardBuild() = 0;
    /**设置loss的种类
     * 
    */
    void SetLossFunction(std::string LossType);
};