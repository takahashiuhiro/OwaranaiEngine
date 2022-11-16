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
        TS* NewTensorFirst = this->SelfCGNode->DerivativeNode->NodeContent->Matmul(this->SelfCGNode->InputNode[1]->NodeContent->T());
        this->SumInput(this->SelfCGNode->InputNode[0], NewTensorFirst);
        TS* NewTensorSecond = this->SelfCGNode->InputNode[0]->NodeContent->T()->Matmul(this->SelfCGNode->DerivativeNode->NodeContent);
        this->SumInput(this->SelfCGNode->InputNode[1], NewTensorSecond);
    }
};