#pragma once
#include "BaseLoss.h"

struct MSELoss:BaseLoss
{
public:
    virtual void LossBuild()
    {
        //建立一个常量-1矩阵和想减掉的矩阵元素乘
        Tensor* Constant = new Tensor(OutputNode[0]->NodeContent->shape, OutputNode[0]->NodeContent->Device, OutputNode[0]->NodeContent->DeviceNum);
        Constant->FillArray(-1.);
        CGNode *ConstantNode = new CGNode(Constant, 0);
        ConstantNode->NodeType["Constant"] = 1;
        CGNode *Intermediate_0 = new CGNode(std::vector<CGNode*>{LabelNode[0], ConstantNode},"EleMul", 1);
        //用输出节点减掉标准节点
        CGNode *Intermediate_1 = new CGNode(std::vector<CGNode*>{OutputNode[0],Intermediate_0},"Add", 1);
        //复制一个节点
        CGNode *Intermediate_2 = new CGNode(std::vector<CGNode*>{Intermediate_1},"Add", 1);
        //两个节点相乘
        CGNode *Intermediate_3 = new CGNode(std::vector<CGNode*>{Intermediate_1, Intermediate_2},"EleMul", 1);
        LossNode = Intermediate_3;
    }
};
