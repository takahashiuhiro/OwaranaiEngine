#pragma once
#include "BaseLoss.h"

struct MSELoss:BaseLoss
{
public:
    virtual void LossBuild()
    {
        //建立一个常量-1矩阵和想减掉的矩阵元素乘
        Tensor* Constant_0 = new Tensor(OutputNode[0]->NodeContent->shape, OutputNode[0]->NodeContent->Device, OutputNode[0]->NodeContent->DeviceNum);
        Constant_0->FillArray(-1.);
        CGNode *ConstantNode_0 = new CGNode(Constant_0, 0);
        ConstantNode_0->NodeType["Constant"] = 1;
        CGNode *Intermediate_0 = new CGNode(std::vector<CGNode*>{LabelNode[0], ConstantNode_0},"Elemul", 1);
        //用输出节点减掉标准节点
        CGNode *Intermediate_1 = new CGNode(std::vector<CGNode*>{OutputNode[0],Intermediate_0},"Add", 1);
        //复制一个节点
        CGNode *Intermediate_2 = new CGNode(std::vector<CGNode*>{Intermediate_1},"Add", 1);
        //两个节点相乘
        CGNode *Intermediate_3 = new CGNode(std::vector<CGNode*>{Intermediate_1, Intermediate_2},"Elemul", 1);
        Hyperparameter FlattenParam;
        FlattenParam.Set("ResDim", HyperparameterTypeConst::SIZET, std::vector<size_t>{3});
        CGNode *Intermediate_4 = new CGNode(std::vector<CGNode*>{Intermediate_3},"Flatten", 1, FlattenParam);
        //求sum的过程
        Tensor* Constant_1 = new Tensor(OutputNode[0]->NodeContent->shape, OutputNode[0]->NodeContent->Device, OutputNode[0]->NodeContent->DeviceNum);
        Constant_1->FillArray(1.);
        while(Constant_1->shape.size() > 3)Constant_1->shape.pop_back();

        size_t FlattenShape = 1;
        for(int a=0;a<Constant_1->shape.size();a++)
        {
            FlattenShape*= Constant_1->shape[a];
        }
        Constant_1->shape[Constant_1->shape.size() - 1]*=Constant_1->ShapeCount/FlattenShape;
        size_t SwapTMP = Constant_1->shape[Constant_1->shape.size() - 1];
        Constant_1->shape[Constant_1->shape.size() - 1] = Constant_1->shape[Constant_1->shape.size() - 2];
        Constant_1->shape[Constant_1->shape.size() - 2] = SwapTMP;

        CGNode *ConstantNode_1 = new CGNode(Constant_1, 0);
        ConstantNode_1->NodeType["Constant"] = 1;

        CGNode *Intermediate_5 = new CGNode(std::vector<CGNode*>{Intermediate_4, ConstantNode_1}, "Matmul", 1);

        LossNode = Intermediate_5;
    }
};
