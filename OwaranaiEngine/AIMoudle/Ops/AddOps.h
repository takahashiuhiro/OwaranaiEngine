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
        this->SelfCGNode->NodeContent = this->SelfCGNode->InputNode[0]->NodeContent->Add(this->SelfCGNode->InputNode[0]->NodeContent);
        for(int a=1 ;a< this->SelfCGNode->InputNode.size();a++)
        {
            this->SelfCGNode->NodeContent = this->SelfCGNode->NodeContent->Add(this->SelfCGNode->InputNode[a]->NodeContent);
        }
        this->SelfCGNode->NodeContent = this->SelfCGNode->NodeContent->Add(this->SelfCGNode->InputNode[0]->NodeContent->MulScalar(-1.));
    }

    virtual void Backward()
    {
        for(int a=0 ;a< this->SelfCGNode->InputNode.size();a++)
        {
            //std::cout<<this->SelfCGNode->InputNode[a]->BackwardBuildFlag<<" shenmeshabi\n";
            //std::cout<<-1<<" shenmeshabi\n";
            if(this->SelfCGNode->InputNode[a]->NeedGradient == 0)continue;
            T* NewCGNode = new T("Add", 1);
            NewCGNode->InputNode.push_back(this->SelfCGNode->DerivativeNode);
            //std::cout<<this->SelfCGNode->InputNode[a]->DerivativeNode->InputNode.size()<<" shenmeshabi\n";
            this->SelfCGNode->InputNode[a]->DerivativeNode->InputNode.push_back(NewCGNode);
            //std::cout<<this->SelfCGNode->InputNode[a]->DerivativeNode<<" meshabi\n";
            //std::cout<<this->SelfCGNode->InputNode[a]->DerivativeNode->InputNode.size()<<" "<<this->SelfCGNode->InputNode[a]<<" shenmeshabiiii\n";
        }
    }
};