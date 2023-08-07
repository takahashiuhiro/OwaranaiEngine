#pragma once
#include "BaseOps.h"
#include <typeinfo>

class EleMulOps: public BaseOps
{
public:
    EleMulOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG);
    ~EleMulOps(){};
    virtual void Forward();
    virtual void Backward();
    /**对first求导.*/
    void BuildSingleGrad(std::string FirstNodeid, std::string SecondNodeid);
};