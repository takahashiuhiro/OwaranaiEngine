#pragma once
#include "StdInclude.h"
#include "../Helpers/MoudleInclude.h"
#include "../TensorCore/MoudleInclude.h"
#include "../Ops/MoudleInclude.h"
#include "../ComputationalGraph/MoudleInclude.h"
#include "../LossFunction/MoudleInclude.h"

struct BaseOptimizer
{
public:

    Hyperparameter OptParams;
    std::vector<CGNode*>InputNodeList;
    std::vector<CGNode*>DerivativeNodeList;
    virtual void UpdateParams() = 0;
};