#pragma once
#include "StdInclude.h"

template<typename T>
struct BaseOps
{
    T* SelfCGNode;
    virtual void Forward() = 0;
    virtual void Backward() = 0;
};
