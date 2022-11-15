#pragma once
#include "StdInclude.h"

template<typename T, typename TS>
struct BaseOps
{
    T* SelfCGNode;
    virtual void Forward() = 0;
    virtual void Backward() = 0;
    virtual void SumInput(T* CGNodePointer, TS* TensorContent)
    {
        if(CGNodePointer->DerivativeNode->NodeContent == nullptr)
        {
            CGNodePointer->DerivativeNode->NodeContent = TensorContent;
        }
        else
        {
            CGNodePointer->DerivativeNode->NodeContent = CGNodePointer->DerivativeNode->NodeContent->Add(TensorContent);
        }
    }
};
