#include "BaseOps.h"
#include "../../CommonDataStructure/Log.h"
#include "../ComputationalGraph.h"

void BaseOps::CommonInit(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG)
{
    this->OpsTypeName = OpsTypeName;
    this->Params = Params;
    CG = ParentCG;
}

void BaseOps::ParamsDefinition()
{
    /**每个输入样本的常数权重.*/
    this->Params.Set("AddWeight", std::make_shared<AddWeightType>());
    /**每个输入样本的是否转置.*/
    this->Params.Set("T", std::make_shared<TType>());
}

std::vector<std::string> & BaseOps::GetInputNodeList()
{
    return this->CG->GetNode(this->Nodeid)->InputNodeidList;
}

std::vector<std::string> & BaseOps::GetInputNodeList(std::string InputNodeid)
{
    return this->CG->GetNode(InputNodeid)->InputNodeidList;
}  

void BaseOps::SetAddWeight(AddWeightType InputNodeWeight)
{
    auto &OutDNodeOpsParamsAddWeight = *(Params.template Get<AddWeightTypePtr>("AddWeight"));
    for(auto& CGNodeidItem:InputNodeWeight)
    {
        OutDNodeOpsParamsAddWeight[CGNodeidItem.first] = CGNodeidItem.second;
    }
}

float BaseOps::GetAddWeight(std::string InputNodeid)
{
    auto &OutDNodeOpsParamsAddWeight = *(Params.template Get<AddWeightTypePtr>("AddWeight"));
    Log::Assert(OutDNodeOpsParamsAddWeight.find(InputNodeid)!=OutDNodeOpsParamsAddWeight.end(), std::string("No Weight id:")+InputNodeid);
    return OutDNodeOpsParamsAddWeight[InputNodeid];
}

void BaseOps::SetT(TType InputNodeIsT)
{
    auto &OutDNodeOpsParamsAddWeight = *(Params.template Get<TTypePtr>("T"));
    for(auto& CGNodeidItem:InputNodeIsT)
    {
        OutDNodeOpsParamsAddWeight[CGNodeidItem.first] = CGNodeidItem.second;
    }
}

bool BaseOps::GetT(std::string InputNodeid)
{
    auto &OutDNodeOpsParamsT = *(Params.template Get<TTypePtr>("T"));
    Log::Assert(OutDNodeOpsParamsT.find(InputNodeid)!=OutDNodeOpsParamsT.end(), std::string("This Node Is Not Set T  Node id:")+InputNodeid);
    return OutDNodeOpsParamsT[InputNodeid];
}
