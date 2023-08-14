#pragma once
#include "BaseOps.h"
#include <typeinfo>

class MatMulOps: public BaseOps
{
public:
    MatMulOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG);
    ~MatMulOps(){};
    virtual void Forward();
    virtual void Backward();
    /**对first求导.*/
    //void BuildSingleGrad(std::string FirstNodeid, std::string SecondNodeid);
};