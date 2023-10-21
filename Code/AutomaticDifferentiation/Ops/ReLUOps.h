#pragma once
#include "BaseOps.h"
#include <typeinfo>

class ReLUOps: public BaseOps
{
public:
    ReLUOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG);
    ~ReLUOps(){};
    virtual void Forward();
    virtual void Backward();
    virtual void AfterSettingShapeComputing();
};
