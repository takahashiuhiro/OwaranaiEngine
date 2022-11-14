#pragma once
#include "BaseOps.h"

template<typename T, typename TS>
struct AddOps:BaseOps<T, TS>
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
        for(int a=0 ;a< this->SelfCGNode->InputNode.size();a++)
        {
            TS* NewTensor = new TS(this->SelfCGNode->DerivativeNode->NodeContent->shape, this->SelfCGNode->DerivativeNode->NodeContent->Device, this->SelfCGNode->DerivativeNode->NodeContent->DeviceNum);
            NewTensor->FillArray(0.);
            NewTensor = NewTensor->Add(this->SelfCGNode->DerivativeNode->NodeContent);
            T* NewCGNode = new T(NewTensor, 1);
            this->SelfCGNode->InputNode[a]->DerivativeNode = NewCGNode;
        }
    }
};