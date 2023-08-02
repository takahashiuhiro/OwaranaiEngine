#pragma once
#include "BaseOps.h"
#include <typeinfo>


//template<typename ComputationalGraph>
class AddOps: public BaseOps//<ComputationalGraph>
{
public:
    AddOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG);

    ~AddOps(){};

    using AddWeightType = std::map<std::string, float>;
    using AddWeightTypePtr = std::shared_ptr<AddWeightType>;

    virtual void Forward();

    virtual void Backward();

    /**在该函数定义Params中的参数.*/
    virtual void ParamsDefinition();

};