#pragma once
#include "BaseOps.h"
#include <typeinfo>

class PowOps: public BaseOps
{
public:
    PowOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG);
    ~PowOps(){};
    virtual void Forward();
    virtual void Backward();
    virtual void AfterSettingShapeComputing();
};
