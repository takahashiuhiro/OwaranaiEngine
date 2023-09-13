#pragma once
#include "BaseOps.h"
#include <typeinfo>

class AddOps: public BaseOps
{
public:
    AddOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG);
    ~AddOps(){};
    virtual void Forward();
    virtual void Backward();
    virtual void AfterSettingShapeComputing();
};