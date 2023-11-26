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

std::string OEAutoDiff::Softmax(ComputationalGraph*CG,std::string InputNode, size_t InputDim)
{
    std::string SoftmaxNodeName = CG->GetNodeidByOps(OpsType::Softmax, {InputNode});
    CG->RegisterVariableNode(SoftmaxNodeName);
    CG->RegisterOpsCompleted(SoftmaxNodeName, {InputNode}, OpsType::Softmax, Dict());
    CG->GetCGOps(SoftmaxNodeName)->SetSelectDim({{InputNode, InputDim}});
    CG->GetCGOps(SoftmaxNodeName)->AfterSettingShapeComputing();
    return SoftmaxNodeName;
}
std::string OEAutoDiff::Softmax(std::shared_ptr<ComputationalGraph>CG,std::string InputNode, size_t InputDim)
{
    return Softmax(CG.get(), InputNode, InputDim);
}

std::string OEAutoDiff::Dropout(ComputationalGraph*CG,std::string InputNode, float P ,bool InPlace)
{
    CG->GetNode(InputNode)->Property.Set("Dropout", true);//是否需要dropout
    CG->GetNode(InputNode)->Property.Set("DropoutP", P);
    return InputNode;
}
std::string OEAutoDiff::Dropout(std::shared_ptr<ComputationalGraph>CG,std::string InputNode,float P ,bool InPlace)
{
    return Dropout(CG.get(),InputNode,P,InPlace);
}

std::string OEAutoDiff::EleExp(ComputationalGraph*CG,std::string InputNode,float BaseNum)
{
    std::string EleExpNodeName = CG->GetNodeidByOps(OpsType::EleExp, {InputNode});
    CG->RegisterVariableNode(EleExpNodeName);
    CG->RegisterOpsCompleted(EleExpNodeName, {InputNode}, OpsType::EleExp, Dict());
    CG->GetCGOps(EleExpNodeName)->SetEleBaseNum(BaseNum);
    CG->GetCGOps(EleExpNodeName)->AfterSettingShapeComputing();
    return EleExpNodeName;
}
std::string OEAutoDiff::EleExp(std::shared_ptr<ComputationalGraph>CG,std::string InputNode,float BaseNum)
{
    return EleExp(CG.get(),InputNode,BaseNum);
}

std::string OEAutoDiff::Tanh(ComputationalGraph*CG,std::string InputNode)
{
    std::string OnesNode = Pow(CG, InputNode, 0.);
    std::string MulSNode = Add(CG,{{InputNode, -2.}});
    std::string EleExpNode = EleExp(CG, MulSNode, M_E);
    std::string OnesMinusNode = Add(CG, {{OnesNode,1.},{EleExpNode,-1.}});
    std::string OnesAddNode = Add(CG, {{OnesNode,1.},{EleExpNode,1.}});
    std::string PowNode = Pow(CG, OnesAddNode, -1.);
    std::string DotNode = EleMul(CG, OnesMinusNode, PowNode);
    return DotNode;
}  
std::string OEAutoDiff::Tanh(std::shared_ptr<ComputationalGraph>CG,std::string InputNode)
{
    return Tanh(CG.get(), InputNode);
}

std::string OEAutoDiff::GELU(ComputationalGraph*CG,std::string InputNode)
{
    std::string PowNode = Pow(CG,InputNode,3.);
    float Kdot = std::pow(2./M_PI,0.5);
    std::string AddInTanhNode = Add(CG, {{InputNode, Kdot},{PowNode,0.044715*Kdot}});
    std::string TanhNode = Tanh(CG, AddInTanhNode);
    std::string DotNode = EleMul(CG, InputNode, TanhNode,0.5,1);
    std::string AddNode = Add(CG, {{InputNode, 0.5}, {DotNode,1}});
    return AddNode;
}
std::string OEAutoDiff::GELU(std::shared_ptr<ComputationalGraph>CG,std::string InputNode)
{
    return GELU(CG.get(), InputNode);
}