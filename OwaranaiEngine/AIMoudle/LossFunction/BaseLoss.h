#pragma once
#include "StdInclude.h"
#include "../Ops/MoudleInclude.h"
#include "../ComputationalGraph/MoudleInclude.h"
#include "../TensorCore/MoudleInclude.h"

struct BaseLoss
{
public:
    std::vector<CGNode*>OutputNode;
    std::vector<CGNode*>LabelNode;
    CGNode* LossNode;
    virtual void LossBuild() = 0;
};
