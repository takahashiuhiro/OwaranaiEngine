#pragma once
#include "BaseOps.h"

template<typename T>
struct AddOps:BaseOps<T>
{
    AddOps(T* SelfCGNode)
    {
        this->SelfCGNode = SelfCGNode;
    }

    virtual void Forward()
    {
        this->SelfCGNode->NodeContent = this->SelfCGNode->InputNode[0]->NodeContent->Add(this->SelfCGNode->InputNode[1]->NodeContent);
    }

    virtual void Backward()
    {
        
    }
};