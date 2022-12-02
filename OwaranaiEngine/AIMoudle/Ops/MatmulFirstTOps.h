#pragma once
#include "BaseOps.h"
/**a T matrix mul to matrix*/
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
        if(this->SelfCGNode->InputNode[0]->NeedGradient)
        {
            T* NewCGNodeFirst = new T("MatmulSecondT", 1);
            NewCGNodeFirst->NodeType["Gradient"] = 1;
            NewCGNodeFirst->InputNode.push_back(this->SelfCGNode->InputNode[1]);
            NewCGNodeFirst->InputNode.push_back(this->SelfCGNode->DerivativeNode);
            this->SelfCGNode->InputNode[0]->DerivativeNode->InputNode.push_back(NewCGNodeFirst);
        }
        if(this->SelfCGNode->InputNode[1]->NeedGradient)
        {
            T* NewCGNodeSecond = new T("MatmulFirstT", 1);
            NewCGNodeSecond->NodeType["Gradient"] = 1;
            NewCGNodeSecond->InputNode.push_back(this->SelfCGNode->InputNode[0]);
            NewCGNodeSecond->InputNode.push_back(this->SelfCGNode->DerivativeNode);
            this->SelfCGNode->InputNode[1]->DerivativeNode->InputNode.push_back(NewCGNodeSecond);
        }  
    }
};