#pragma once
#include "BaseOps.h"
#include <typeinfo>

class SumOps: public BaseOps
{
public:
    SumOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG);
    ~SumOps(){};
    virtual void Forward();
    virtual void Backward();
    virtual void AfterSettingShapeComputing();
};
