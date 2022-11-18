#pragma once
#include "BaseOps.h"

template<typename T, typename TS>
struct MatmulOps:BaseOps<T, TS>
{
    MatmulOps(T* SelfCGNode)
    {
        this->SelfCGNode = SelfCGNode;
    }

    virtual void Forward()
    {
        this->SelfCGNode->NodeContent = this->SelfCGNode->InputNode[0]->NodeContent->Matmul(this->SelfCGNode->InputNode[1]->NodeContent);
    }

    virtual void Backward()
    {

    }
};