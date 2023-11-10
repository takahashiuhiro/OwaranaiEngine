#include "ForwardFunction.h"
#include "ComputationalGraph.h"

std::string OEAutoDiff::Add(ComputationalGraph*CG,std::map<std::string, float> InputWeight)
{
    std::vector<std::string> InputNodes;
    for(auto & InputPair:InputWeight)InputNodes.push_back(InputPair.first);
    std::string AddNode = CG->GetNodeidByOps(OpsType::Add, InputNodes);
    CG->RegisterVariableNode(AddNode);
    CG->RegisterOpsCompleted(AddNode, InputNodes, OpsType::Add, Dict());
    CG->GetCGOps(AddNode)->SetAddWeight(InputWeight);
    CG->GetCGOps(AddNode)->AfterSettingShapeComputing();
    return AddNode;
}
std::string OEAutoDiff::Add(std::shared_ptr<ComputationalGraph>CG,std::map<std::string, float> InputWeight)
{
    return Add(CG.get(), InputWeight);
}

std::string OEAutoDiff::Pow(ComputationalGraph*CG,std::string InputNode,float Exponent)
{
    std::string PowNode = CG->GetNodeidByOps(OpsType::Pow, {InputNode});
    CG->RegisterVariableNode(PowNode);
    CG->RegisterOpsCompleted(PowNode, {InputNode}, OpsType::Pow, Dict());
    CG->GetCGOps(PowNode)->SetEleExponent(Exponent);
    CG->GetCGOps(PowNode)->AfterSettingShapeComputing();
    return PowNode;
}
std::string OEAutoDiff::Pow(std::shared_ptr<ComputationalGraph>CG,std::string InputNode,float Exponent)
{
    return Pow(CG.get(), InputNode, Exponent);
}

std::string OEAutoDiff::BroadCastTo(ComputationalGraph*CG,std::string InputNode,std::vector<size_t>InputDims)
{
    std::string BCNode = CG->GetNodeidByOps(OpsType::BroadCastTo, {InputNode});
    CG->RegisterVariableNode(BCNode);
    CG->RegisterOpsCompleted(BCNode, {InputNode}, OpsType::BroadCastTo, Dict());
    CG->GetCGOps(BCNode)->SetBroadCastTo({{InputNode, InputDims}});
    CG->GetCGOps(BCNode)->AfterSettingShapeComputing();
    return BCNode;
}
std::string OEAutoDiff::BroadCastTo(std::shared_ptr<ComputationalGraph>CG,std::string InputNode,std::vector<size_t>InputDims)
{
    return BroadCastTo(CG.get(), InputNode, InputDims);
}

std::string OEAutoDiff::EleMul(ComputationalGraph*CG,std::string FirstNode, std::string SecondNode,float FirstAddWeight, float SecondAddWeight)
{
    std::vector<std::string> InputNodes = {FirstNode, SecondNode};
    std::string EleMulNode = CG->GetNodeidByOps(OpsType::EleMul, InputNodes);
    CG->RegisterVariableNode(EleMulNode);
    CG->RegisterOpsCompleted(EleMulNode, InputNodes, OpsType::EleMul, Dict());
    CG->GetCGOps(EleMulNode)->SetAddWeight({{FirstNode, FirstAddWeight}, {SecondNode, SecondAddWeight}});
    CG->GetCGOps(EleMulNode)->AfterSettingShapeComputing();
    return EleMulNode;
}
std::string OEAutoDiff::EleMul(std::shared_ptr<ComputationalGraph>CG,std::string FirstNode, std::string SecondNode,float FirstAddWeight, float SecondAddWeight)
{
    return EleMul(CG.get(), FirstNode,SecondNode,FirstAddWeight,SecondAddWeight);
}

std::string OEAutoDiff::MatMul(ComputationalGraph*CG, std::string FirstNode, std::string SecondNode, bool FirstTFlag, bool SecondTFlag, float FirstAddWeight, float SecondAddWeight)
{
    std::string MatMulNodeName = CG->GetNodeidByOps(OpsType::MatMul, {FirstNode, SecondNode});
    CG->RegisterVariableNode(MatMulNodeName);
    CG->RegisterOpsCompleted(MatMulNodeName, {FirstNode, SecondNode}, OpsType::MatMul, Dict());
    CG->GetCGOps(MatMulNodeName)->SetT({{FirstNode, FirstTFlag},{SecondNode, SecondTFlag}});
    CG->GetCGOps(MatMulNodeName)->SetAddWeight({{FirstNode, FirstAddWeight},{SecondNode, SecondAddWeight}});
    CG->GetCGOps(MatMulNodeName)->AfterSettingShapeComputing();
    return MatMulNodeName;
}
std::string OEAutoDiff::MatMul(std::shared_ptr<ComputationalGraph>CG,std::string FirstNode, std::string SecondNode, bool FirstTFlag, bool SecondTFlag, float FirstAddWeight, float SecondAddWeight)
{
    return MatMul(CG.get(),FirstNode,SecondNode, FirstTFlag,SecondTFlag, FirstAddWeight, SecondAddWeight);
}

std::string OEAutoDiff::Sum(ComputationalGraph*CG,std::string InputNode, std::vector<size_t>InputDims)
{
    std::string SumNodeName = CG->GetNodeidByOps(OpsType::Sum, {InputNode});
    CG->RegisterVariableNode(SumNodeName);
    CG->RegisterOpsCompleted(SumNodeName, {InputNode}, OpsType::Sum, Dict());
    CG->GetCGOps(SumNodeName)->SetSelectDims({{InputNode, InputDims}});
    CG->GetCGOps(SumNodeName)->AfterSettingShapeComputing();
    return SumNodeName;
}
std::string OEAutoDiff::Sum(std::shared_ptr<ComputationalGraph>CG,std::string InputNode, std::vector<size_t>InputDims)
{
    return Sum(CG.get(), InputNode, InputDims);
}

std::string OEAutoDiff::Mean(ComputationalGraph*CG,std::string InputNode, std::vector<size_t>InputDims)
{
    float SumDimRes = 1;
    for(size_t a = 0;a<InputDims.size();a++)
    {
        SumDimRes*=CG->GetNode(InputNode)->NodeContentShape[InputDims[a]];
    }
    SumDimRes = 1./SumDimRes;
    std::string SumNodeName = Sum(CG, InputNode, InputDims);
    std::string NewNodeName = Add(CG, {{SumNodeName,SumDimRes}});
    return NewNodeName;
}

std::string OEAutoDiff::Mean(std::shared_ptr<ComputationalGraph>CG,std::string InputNode, std::vector<size_t>InputDims)
{
    return Mean(CG.get(),InputNode, InputDims);
}

std::string OEAutoDiff::Var(ComputationalGraph*CG,std::string InputNode, std::vector<size_t>InputDims, bool Unbiased)
{
    std::string MeanSlimNode = Mean(CG, InputNode, InputDims);
    std::string BCNode = BroadCastTo(CG,MeanSlimNode,CG->GetNode(InputNode)->NodeContentShape);
    std::string MinusNode = Add(CG, {{InputNode,1.},{BCNode,-1.}});
    std::string PowNode = Pow(CG, MinusNode, 2.);
    std::string SumNode = Sum(CG, PowNode, InputDims);
    float SumDimRes = 1;
    for(size_t a = 0;a<InputDims.size();a++)
    {
        SumDimRes*=CG->GetNode(InputNode)->NodeContentShape[InputDims[a]];
    }
    SumDimRes -= Unbiased;
    SumDimRes = 1./SumDimRes;
    std::string DivideNode = Add(CG, {{SumNode,SumDimRes}});
    return DivideNode;
}

std::string OEAutoDiff::Var(std::shared_ptr<ComputationalGraph>CG,std::string InputNode, std::vector<size_t>InputDims, bool Unbiased)
{
    return Var(CG.get(),InputNode, InputDims,Unbiased);
}

std::string OEAutoDiff::ReLU(ComputationalGraph*CG,std::string InputNode)
{
    std::string ReLUNodeName = CG->GetNodeidByOps(OpsType::ReLU, {InputNode});
    CG->RegisterVariableNode(ReLUNodeName);
    CG->RegisterOpsCompleted(ReLUNodeName, {InputNode}, OpsType::ReLU, Dict());
    CG->GetCGOps(ReLUNodeName)->AfterSettingShapeComputing();
    return ReLUNodeName;
}
std::string OEAutoDiff::ReLU(std::shared_ptr<ComputationalGraph>CG,std::string InputNode)
{
    return ReLU(CG.get(), InputNode);
}