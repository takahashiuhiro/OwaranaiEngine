#pragma once
#include "BaseOps.h"
/**elemul*/
template<typename T, typename TS>
struct ElemulOps:BaseOps<T, TS>
{
    ElemulOps(T* SelfCGNode)
    {
        this->SelfCGNode = SelfCGNode;
    }

    virtual void Forward()
    {
        this->SelfCGNode->NodeContent = this->SelfCGNode->InputNode[0]->NodeContent->EleMul(this->SelfCGNode->InputNode[1]->NodeContent);
    }

    virtual void Backward()
    {
        if(this->SelfCGNode->InputNode[0]->NeedGradient)
        {
            T* NewCGNodeFirst = new T("Elemul", 1);
            NewCGNodeFirst->NodeType["Gradient"] = 1;
            NewCGNodeFirst->InputNode.push_back(this->SelfCGNode->DerivativeNode);
            NewCGNodeFirst->InputNode.push_back(this->SelfCGNode->InputNode[1]);
            this->SelfCGNode->InputNode[0]->DerivativeNode->InputNode.push_back(NewCGNodeFirst);
        }
        if(this->SelfCGNode->InputNode[1]->NeedGradient)
        {
            T* NewCGNodeSecond = new T("Elemul", 1);
            NewCGNodeSecond->NodeType["Gradient"] = 1;
            NewCGNodeSecond->InputNode.push_back(this->SelfCGNode->InputNode[0]);
            NewCGNodeSecond->InputNode.push_back(this->SelfCGNode->DerivativeNode);
            this->SelfCGNode->InputNode[1]->DerivativeNode->InputNode.push_back(NewCGNodeSecond);
        }        
    }
};