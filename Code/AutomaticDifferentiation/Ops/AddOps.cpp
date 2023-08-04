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
    if(NodeidList.size()== 0)return;//暂定加算子没有输入直接退出
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
        auto InputNode = this->CG->GetNode(NodeidList[a]);
        if(InputNode->Property.Get<bool>("RequireGrad") == false)
        {
            //确定打了不需要导数的标就不需要建立链接
            continue;
        }
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
    BaseOps::ParamsDefinition();
}