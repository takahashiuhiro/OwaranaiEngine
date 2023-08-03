#include "AddOps.h"
#include "../ComputationalGraph.h"

AddOps::AddOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG)
{
    BaseOps::CommonInit(OpsTypeName, Params, ParentCG);
    ParamsDefinition();
}

void AddOps::Forward()
{
    std::vector<std::string> &NodeidList = this->CG->GetNode(this->Nodeid)->InputNodeidList;
    Tensor* FirstInputNode = this->CG->GetNode(NodeidList[0])->GetContent();
    Tensor* NodeRes = new Tensor(FirstInputNode->shape, FirstInputNode->GetDeviceNum());
    NodeRes->FillArray(0);
    auto &OutDNodeOpsParamsAddWeight = *(this->Params.template Get<AddWeightTypePtr>("AddWeight"));
    for(size_t a = 0;a<NodeidList.size();a++)
    {
        std::shared_ptr<Tensor> EachContentMulWeight = std::shared_ptr<Tensor>(this->CG->GetNode(NodeidList[a])->GetContent()->MulScalar(OutDNodeOpsParamsAddWeight[NodeidList[a]]));
        NodeRes = NodeRes->Add(EachContentMulWeight.get());
    }
    this->CG->GetNode(this->Nodeid)->AssignContent(NodeRes);
}

void AddOps::Backward()
{
    std::vector<std::string> &NodeidList = this->CG->GetNode(this->Nodeid)->InputNodeidList;
    std::string ThisDNodeid = this->CG->GetDNodeid(this->Nodeid);
    for(size_t a=0;a<NodeidList.size();a++)
    {
        std::string InputDNodeid = this->CG->GetDNodeid(NodeidList[a]);
        this->CG->RegisterOpsAddEdge(InputDNodeid, ThisDNodeid);
        auto &InputDNodeOpsParams = this->CG->Opss[InputDNodeid]->Params;
        auto &InputDNodeOpsParamsAddWeight = *(InputDNodeOpsParams.template Get<AddWeightTypePtr>("AddWeight"));
        auto &OutDNodeOpsParamsAddWeight = *(this->Params.template Get<AddWeightTypePtr>("AddWeight"));
        InputDNodeOpsParamsAddWeight[ThisDNodeid] = OutDNodeOpsParamsAddWeight[NodeidList[a]];
    }
}

void AddOps::ParamsDefinition()
{
    /**每个输入样本的常数权重.*/
    this->Params.Set("AddWeight", std::make_shared<AddWeightType>());
}