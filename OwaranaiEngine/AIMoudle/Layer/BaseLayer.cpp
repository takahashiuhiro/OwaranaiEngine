#include "BaseLayer.h"

void BaseLayer::LayerInit(std::vector<CGNode*>InputCGNode)
{
    this->InputCGNode = InputCGNode;
}

void BaseLayer::Freeze()
{
    for(int a =0;a<ParamsCGNode.size();a++)
    {
        ParamsCGNode[a]->NodeType["Freeze"] = 1;
    }
}

CGNode* BaseLayer::Forward()
{
    ForwardNode->Forward();
    return ForwardNode;
}

void BaseLayer::Backward(Tensor* Loss)
{
    ForwardNode->Backward(Loss);
    for(int a=0;a<ParamsCGNode.size();a++)
    {
        //ParamsCGNode[a]->DerivativeNode->Forward();
        std::cout<<ParamsCGNode[a]->DerivativeNode<<"  output"<<std::endl;
        /** todo::这里有点问题，在执行了backward后有计算节点没被构造导数节点，会导致段错误...打印如下
         * ------------------------GPU test---------------------------------
         * 0x5587188f1d50  output
         * 0  output
        */
    }
}
