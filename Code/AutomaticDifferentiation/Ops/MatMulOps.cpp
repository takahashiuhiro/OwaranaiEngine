#include "MatMulOps.h"
#include "../ComputationalGraph.h"

MatMulOps::MatMulOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG)
{
    BaseOps::CommonInit(OpsTypeName, Params, ParentCG);
    ParamsDefinition();
}

void MatMulOps::Forward()
{
    auto NodeidList = GetInputNodeList();
    Log::Assert(NodeidList.size() == 2, std::string("MatmulOps Must Have 2 Input Node"));
    Tensor* FirstInputNode = this->CG->GetNode(NodeidList[0])->GetContent();
    Tensor* SecondInputNode = this->CG->GetNode(NodeidList[1])->GetContent();
    bool TFlagFirst = GetT(NodeidList[0]);
    bool TFlagSecond = GetT(NodeidList[1]);
    if(TFlagFirst)FirstInputNode = FirstInputNode->T();
    if(TFlagSecond)SecondInputNode = SecondInputNode->T();
    float AddWeightDot = GetAddWeight(NodeidList[0])*GetAddWeight(NodeidList[1]);
    std::shared_ptr<Tensor> NodeRes = std::shared_ptr<Tensor>(FirstInputNode->Matmul(SecondInputNode));
    this->CG->GetNode(this->Nodeid)->AssignContent(NodeRes->MulScalar(AddWeightDot));
    if(TFlagFirst)delete FirstInputNode;
    if(TFlagSecond)delete SecondInputNode;
}

void MatMulOps::Backward()
{
    auto NodeidList = GetInputNodeList();
    Log::Assert(NodeidList.size() == 2, std::string("MatmulOps Must Have 2 Input Node"));
    bool TFlagFirst = GetT(NodeidList[0]);
    bool TFlagSecond = GetT(NodeidList[1]);
    float AddWeightDot = GetAddWeight(NodeidList[0])*GetAddWeight(NodeidList[1]);
    //bug::这里没考虑一开始就俩都有TFLAG应该怎么办
    if(this->CG->GetNode(NodeidList[0])->Property.Get<bool>("RequireGrad"))
    {
        std::string NewDNode;
        if(TFlagFirst == false&&TFlagSecond==false)NewDNode = OEAutoDiff::MatMul(this->CG, this->CG->GetDNodeid(Nodeid), NodeidList[1], false, true, 1., AddWeightDot);
        if(TFlagFirst == false&&TFlagSecond==true)NewDNode = OEAutoDiff::MatMul(this->CG, this->CG->GetDNodeid(Nodeid), NodeidList[1], false, false, 1., AddWeightDot);
        if(TFlagFirst == true&&TFlagSecond==false)NewDNode = OEAutoDiff::MatMul(this->CG,NodeidList[1], this->CG->GetDNodeid(Nodeid), false, true, 1., AddWeightDot);
        if(TFlagFirst == true&&TFlagSecond==true)NewDNode = OEAutoDiff::MatMul(this->CG, NodeidList[1],this->CG->GetDNodeid(Nodeid),true,true,1.,AddWeightDot);
        this->CG->RegisterOpsAddEdge(this->CG->GetDNodeid(NodeidList[0]), NewDNode);
        this->CG->GetCGOps(this->CG->GetDNodeid(NodeidList[0]))->SetAddWeight({{NewDNode, 1.}});
    }
    if(this->CG->GetNode(NodeidList[1])->Property.Get<bool>("RequireGrad"))
    {
        std::string NewDNode;
        if(TFlagFirst == false&&TFlagSecond==false)NewDNode = OEAutoDiff::MatMul(this->CG,NodeidList[0], this->CG->GetDNodeid(Nodeid),true, false, 1., AddWeightDot);
        if(TFlagFirst == false&&TFlagSecond==true)NewDNode = OEAutoDiff::MatMul(this->CG, this->CG->GetDNodeid(Nodeid), NodeidList[0], true, false, 1., AddWeightDot);
        if(TFlagFirst == true&&TFlagSecond==false)NewDNode = OEAutoDiff::MatMul(this->CG, NodeidList[0],this->CG->GetDNodeid(Nodeid), false, false, 1., AddWeightDot);
        if(TFlagFirst == true&&TFlagSecond==true)NewDNode = OEAutoDiff::MatMul(this->CG,this->CG->GetDNodeid(Nodeid) ,NodeidList[0],true,true,1.,AddWeightDot);
        this->CG->RegisterOpsAddEdge(this->CG->GetDNodeid(NodeidList[1]), NewDNode);
        this->CG->GetCGOps(this->CG->GetDNodeid(NodeidList[1]))->SetAddWeight({{NewDNode, 1.}});
    }
}

void MatMulOps::AfterSettingShapeComputing()
{
    auto NodeidList = GetInputNodeList();
    this->CG->GetNode(this->Nodeid)->NodeContentShape = this->CG->GetNode(NodeidList[0])->NodeContentShape;
    size_t LastIndex = this->CG->GetNode(this->Nodeid)->NodeContentShape.size()-1;
    bool TFlagFirst = GetT(NodeidList[0]);
    bool TFlagSecond = GetT(NodeidList[1]);
    if(TFlagFirst)this->CG->GetNode(this->Nodeid)->NodeContentShape[LastIndex-1] = this->CG->GetNode(NodeidList[0])->NodeContentShape[LastIndex];
    else this->CG->GetNode(this->Nodeid)->NodeContentShape[LastIndex-1] = this->CG->GetNode(NodeidList[0])->NodeContentShape[LastIndex-1];
    if(TFlagSecond)this->CG->GetNode(this->Nodeid)->NodeContentShape[LastIndex] = this->CG->GetNode(NodeidList[1])->NodeContentShape[LastIndex-1];
    else this->CG->GetNode(this->Nodeid)->NodeContentShape[LastIndex] = this->CG->GetNode(NodeidList[1])->NodeContentShape[LastIndex];
}