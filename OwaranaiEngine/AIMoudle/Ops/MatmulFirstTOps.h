#pragma once
#include "BaseOps.h"

template<typename T, typename TS>
struct MatmulFirstTOps:BaseOps<T, TS>
{
    MatmulFirstTOps(T* SelfCGNode)
    {
        this->SelfCGNode = SelfCGNode;
    }

    virtual void Forward()
    {
        this->SelfCGNode->NodeContent = this->SelfCGNode->InputNode[0]->NodeContent->T()->Matmul(this->SelfCGNode->InputNode[1]->NodeContent);
    }

    virtual void Backward()
    {

    }
};