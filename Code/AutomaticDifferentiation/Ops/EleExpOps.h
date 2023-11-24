#pragma once
#include "BaseOps.h"
#include <typeinfo>

class EleExpOps: public BaseOps
{
public:
    EleExpOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG);
    ~EleExpOps(){};
    virtual void Forward();
    virtual void Backward();
    virtual void AfterSettingShapeComputing();
};