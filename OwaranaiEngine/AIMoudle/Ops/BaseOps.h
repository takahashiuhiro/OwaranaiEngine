#pragma once
#include "StdInclude.h"
#include "../Helpers/MoudleInclude.h"

template<typename T, typename TS>
struct BaseOps
{
    T* SelfCGNode;
    Hyperparameter Params;
    virtual void Forward() = 0;
    virtual void Backward() = 0;
};
