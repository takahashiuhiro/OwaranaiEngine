#pragma once
#include "BaseOps.h"

template<typename T, typename TS>
struct MatmulSecondTOps:BaseOps<T, TS>
{
    MatmulSecondTOps(T* SelfCGNode)
    {
        this->SelfCGNode = SelfCGNode;
    }

    virtual void Forward()
    {
        this->SelfCGNode->NodeContent = this->SelfCGNode->InputNode[0]->NodeContent->Matmul(this->SelfCGNode->InputNode[1]->NodeContent->T());
    }

    virtual void Backward()
    {
        
    }
};