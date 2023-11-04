#include "ForwardFunction.h"
#include "ComputationalGraph.h"

std::vector<std::string> OEAutoDiff::Mean(ComputationalGraph*CG,std::vector<std::string>InputNodes, std::vector<size_t>InputDims)
{
    auto Node = CG->GetNode(InputNodes[0]);
    float SumDimRes = 1;
    for(size_t a = 0;a<InputDims.size();a++)
    {
        SumDimRes*=Node->NodeContentShape[InputDims[a]];
    }
    SumDimRes = 1./SumDimRes;
    std::string SumNodeName = CG->GetNodeidByOps(OpsType::Sum, InputNodes);
    CG->RegisterVariableNode(SumNodeName);
    CG->RegisterOpsCompleted(SumNodeName, InputNodes, OpsType::Sum, Dict());
    CG->GetCGOps(SumNodeName)->SetSelectDims({{InputNodes[0], InputDims}});
    CG->GetCGOps(SumNodeName)->AfterSettingShapeComputing();
    std::string NewNodeName = CG->GetNodeidByOps(OpsType::Add, {SumNodeName});
    CG->RegisterVariableNode(NewNodeName);
    CG->RegisterOpsCompleted(NewNodeName, {SumNodeName}, OpsType::Add, Dict());
    CG->GetCGOps(NewNodeName)->SetAddWeight({{SumNodeName, SumDimRes}});
    CG->GetCGOps(NewNodeName)->AfterSettingShapeComputing();
    return {NewNodeName};
}

std::vector<std::string> OEAutoDiff::Mean(std::shared_ptr<ComputationalGraph>CG,std::vector<std::string>InputNodes, std::vector<size_t>InputDims)
{
    return Mean(CG.get(),InputNodes, InputDims);
}
