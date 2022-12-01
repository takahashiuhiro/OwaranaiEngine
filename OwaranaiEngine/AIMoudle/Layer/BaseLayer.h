#pragma once
#include "StdInclude.h"
#include "../TensorCore/MoudleInclude.h"
#include "../Ops/MoudleInclude.h"
#include "../ComputationalGraph/MoudleInclude.h"

struct BaseLayer
{
public:

    /**input list of layer*/
    std::vector<CGNode*>InputCGNode;
    /**param matrix list*/
    std::vector<CGNode*>ParamsCGNode;
    /**forward output*/
    CGNode* ForwardNode;
    /**init by InputCGNode list*/
    void LayerInit(std::vector<CGNode*>InputCGNode);
    /**get the result of the episode*/
    CGNode* Forward();
    void Backward(Tensor* Loss);
    //todo::还缺个能清理梯度数据的玩意，应该就差不多了
    /**do not update this layer's params matrix*/
    void Freeze();
    /**一个虚函数，layer类必须得重写这玩意，在里面指定输出节点构建网络啥的，不装了，我这英语确实不会描述这玩意了...*/
    virtual void ForwardBuild() = 0;
    virtual void ClearGrandint() = 0;
};