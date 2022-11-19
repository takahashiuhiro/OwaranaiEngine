#pragma once
#include "BaseOps.h"
/**a matrix mul to a T matrix*/
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
        if(this->SelfCGNode->InputNode[0]->NeedGradient)
        {
            T* NewCGNodeFirst = new T("MatmulSecondT", 1);
            NewCGNodeFirst->InputNode.push_back(this->SelfCGNode->DerivativeNode);
            NewCGNodeFirst->InputNode.push_back(this->SelfCGNode->InputNode[1]);
            this->SelfCGNode->InputNode[0]->DerivativeNode->InputNode.push_back(NewCGNodeFirst);
        }
        if(this->SelfCGNode->InputNode[1]->NeedGradient)
        {
            T* NewCGNodeSecond = new T("MatmulFirstT", 1);
            NewCGNodeSecond->InputNode.push_back(this->SelfCGNode->DerivativeNode);
            NewCGNodeSecond->InputNode.push_back(this->SelfCGNode->InputNode[0]);
            this->SelfCGNode->InputNode[1]->DerivativeNode->InputNode.push_back(NewCGNodeSecond);
        }   
    }
};