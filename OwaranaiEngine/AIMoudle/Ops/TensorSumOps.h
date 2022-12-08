#pragma once
#include "BaseOps.h"
/**make >0 matrix add*/
template<typename T, typename TS>
struct TensorSumOps:BaseOps<T, TS>
{
    TensorSumOps(T* SelfCGNode)
    {
        this->SelfCGNode = SelfCGNode;
    }
    virtual void Forward()
    {
        //前向嗯+就行了，其实没什么所谓的...
    }

    virtual void Backward()
    {
        //关键是反向只能搭建节点，这个反向对应的前向要怎么写...
        //有点想法，比如反向搭建一个叫首元素乘的东西...用一边的乘法去乘另一边，可是这玩意又得写反向，这又得怎么写啊...
        //这样下去没完没了，是不是得考虑一下...一个算子是否允许他没有反向呢?
    }
};