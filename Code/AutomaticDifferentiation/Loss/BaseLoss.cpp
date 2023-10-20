#include "BaseLoss.h"

void BaseLoss::CommonInit(std::shared_ptr<ComputationalGraph>InputCG)
{
    CG = InputCG;
}

void BaseLoss::SetLossData(std::map<std::string, Tensor*>LossMap)
{
    for(auto& LossPair:LossMap)
    {
        CG->GetNode(LossPair.first)->AssignContent(LossPair.second);
    }
}

void BaseLoss::Backward()
{
    CG->ForwardDfs(LossNodes[0]);//计算图执行到loss节点
    CG->GetNode(CG->GetDNodeid(LossNodes[0]))->AssignContent(CG->GetNode(LossNodes[0])->GetContent()->Copy());//把loss的结果赋值给他的梯度节点
    CG->ComputeWeightNodesDForward();//执行计算图内所有允许求导的节点的反向
}