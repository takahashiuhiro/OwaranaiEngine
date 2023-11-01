#pragma once
#include "BaseOps.h"
#include <typeinfo>

class MeanOps: public BaseOps
{
public:
    MeanOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG);
    ~MeanOps(){};
    virtual void Forward();
    virtual void Backward();
    virtual void AfterSettingShapeComputing();
};