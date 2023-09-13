#pragma once
#include "BaseOps.h"
#include <typeinfo>

class SoftmaxOps: public BaseOps
{
public:
    SoftmaxOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG);
    ~SoftmaxOps(){};
    virtual void Forward();
    virtual void Backward();
    virtual void AfterSettingShapeComputing();
};
