#pragma once
#include "BaseOps.h"
/**只对内里元素处理的add算子*/
template<typename T, typename TS>
struct AddarrayOps:BaseOps<T, TS>
{
    AddarrayOps(T* SelfCGNode)
    {
        this->SelfCGNode = SelfCGNode;
    }

    virtual void Forward()
    {
        this->SelfCGNode->NodeContent = this->SelfCGNode->InputNode[0]->NodeContent->AddArray(this->SelfCGNode->InputNode[0]->NodeContent);
        for(int a=1 ;a< this->SelfCGNode->InputNode.size();a++)
        {
            this->SelfCGNode->NodeContent = this->SelfCGNode->NodeContent->AddArray(this->SelfCGNode->InputNode[a]->NodeContent);
        }
        this->SelfCGNode->NodeContent = this->SelfCGNode->NodeContent->AddArray(this->SelfCGNode->InputNode[0]->NodeContent->MulScalar(-1.));
    }

    virtual void Backward()
    {
        for(int a=0 ;a< this->SelfCGNode->InputNode.size();a++)
        {
            if(this->SelfCGNode->InputNode[a]->NeedGradient == 0)continue;
            T* NewCGNode = new T("AddArray", 1);
            NewCGNode->NodeType["Gradient"] = 1;
            NewCGNode->InputNode.push_back(this->SelfCGNode->DerivativeNode);
            this->SelfCGNode->InputNode[a]->DerivativeNode->InputNode.push_back(NewCGNode);
        }
    }
};