#pragma once
#include "BaseOps.h"
#include <typeinfo>

class GenerateSignOps: public BaseOps
{
public:
    GenerateSignOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG);
    ~GenerateSignOps(){};
    virtual void Forward();
    virtual void AfterSettingShapeComputing();
};
