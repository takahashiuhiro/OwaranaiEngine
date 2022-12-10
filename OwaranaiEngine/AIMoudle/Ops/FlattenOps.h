#pragma once
#include "BaseOps.h"
/**make >0 matrix sum*/
template<typename T, typename TS>
struct FlattenOps:BaseOps<T, TS>
{
    FlattenOps(T* SelfCGNode, Hyperparameter InputParams)
    {
        //应该填从左到右留了几位
        this->SelfCGNode = SelfCGNode;
        this->Params = InputParams;
    }
    virtual void Forward()
    {
        this->SelfCGNode->NodeContent = this->SelfCGNode->InputNode[0]->NodeContent->AddArray(this->SelfCGNode->InputNode[0]->NodeContent);
        this->SelfCGNode->NodeContent = this->SelfCGNode->NodeContent->AddArray(this->SelfCGNode->InputNode[0]->NodeContent->MulScalar(-1.));
        size_t ResDim = (*(this->Params).Get("ResDim"))[0];
        while(this->SelfCGNode->NodeContent->shape.size() > ResDim)
        {
            this->SelfCGNode->NodeContent->shape.pop_back();
        }
    }

    virtual void Backward()
    {
        for(int a=0 ;a< this->SelfCGNode->InputNode.size();a++)
        {
            if(this->SelfCGNode->InputNode[a]->NeedGradient == 0)continue;
            T* NewCGNode = new T("Addarray", 1);
            NewCGNode->NodeType["Gradient"] = 1;
            NewCGNode->InputNode.push_back(this->SelfCGNode->DerivativeNode);
            this->SelfCGNode->InputNode[a]->DerivativeNode->InputNode.push_back(NewCGNode);
        }
    }
};