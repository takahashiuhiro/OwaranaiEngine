#pragma once
#include "BaseOps.h"
#include <typeinfo>

class ViewOps: public BaseOps
{
public:
    ViewOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG);
    ~ViewOps(){};
    virtual void Forward();
    virtual void Backward();
    virtual void AfterSettingShapeComputing();
};
