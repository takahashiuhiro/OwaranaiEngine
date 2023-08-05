#include "BaseOps.h"
#include "../../CommonDataStructure/Log.h"

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
}

void BaseOps::SetAddWeight(std::map<std::string, float> InputNodeWeight)
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