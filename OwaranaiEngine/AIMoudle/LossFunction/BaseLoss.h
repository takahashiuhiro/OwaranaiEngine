#pragma once
#include "StdInclude.h"
#include "../Ops/MoudleInclude.h"
#include "../ComputationalGraph/MoudleInclude.h"
#include "../../CommonMathMoudle/MoudleInclude.h"
#include "../Helpers/MoudleInclude.h"

struct BaseLoss
{
public:
    std::vector<CGNode*>OutputNode;
    std::vector<CGNode*>LabelNode;
    CGNode* LossNode;
    Hyperparameter LossParams;
    virtual void LossBuild() = 0;
};
