#pragma once
#include "BaseOps.h"
#include <typeinfo>


//template<typename ComputationalGraph>
class AddOps: public BaseOps//<ComputationalGraph>
{
public:
    AddOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG);
    ~AddOps(){};

    virtual void Forward();

    virtual void Backward();

    /**在该函数定义Params中的参数.*/
    virtual void ParamsDefinition();

};