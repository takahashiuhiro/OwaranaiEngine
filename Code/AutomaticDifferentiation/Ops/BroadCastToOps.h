#pragma once
#include "BaseOps.h"
#include <typeinfo>

class BroadCastToOps: public BaseOps
{
public:
    BroadCastToOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG);
    ~BroadCastToOps(){};
    virtual void Forward();
    virtual void Backward();
    virtual void AfterSettingShapeComputing();
};

