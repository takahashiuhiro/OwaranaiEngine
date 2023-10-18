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