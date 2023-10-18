#include "MSELoss.h"

void MSELoss::Build(std::vector<std::string>InputCGNodeList, std::vector<std::string>LabelNodeList)
{
    std::vector<std::string> InputVector = {InputCGNodeList[0], LabelNodeList[0]};
    std::string NewDNode = CG->GetNodeidByOps(OpsType::Add, InputVector);
    CG->RegisterVariableNode(NewDNode);
    CG->RegisterOpsCompleted(NewDNode, InputVector, OpsType::Add, Dict());
    CG->GetCGOps(NewDNode)->SetAddWeight({{LabelNodeList[0],-1.}});
    std::string NewDNodeCopy = CG->GetCopyNode(NewDNode);
    CG->RegisterVariableNode(NewDNodeCopy);
    CG->RegisterOpsCompleted(NewDNodeCopy, {NewDNode}, OpsType::Add, Dict());
    std::string DotNode = CG->GetNodeidByOps(OpsType::EleMul, {NewDNode, NewDNodeCopy});
    CG->RegisterVariableNode(DotNode);
    CG->RegisterOpsCompleted(DotNode, {NewDNode, NewDNodeCopy}, OpsType::EleMul, Dict());
    LossNodes.push_back(DotNode);
}