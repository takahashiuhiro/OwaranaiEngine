#pragma once
#include "BaseOps.h"
#include <typeinfo>

template<typename ComputationalGraph>
class AddOps: public BaseOps<ComputationalGraph>
{
public:
    AddOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG)
    {
        BaseOps<ComputationalGraph>::CommonInit(OpsTypeName, Params, ParentCG);
        ParamsDefinition();
    }

    ~AddOps(){};

    using AddWeightType = std::map<std::string, float>;
    using AddWeightTypePtr = std::shared_ptr<AddWeightType>;

    virtual void Forward()
    {
        std::vector<std::string> &NodeidList = this->CG->GetNode(this->Nodeid)->InputNodeidList;
        Tensor* FirstInputNode = this->CG->GetNode(NodeidList[0])->Content;
        Tensor* NodeRes = new Tensor(FirstInputNode->shape, FirstInputNode->GetDeviceNum());
        NodeRes->FillArray(0);
        for(size_t a = 0;a<NodeidList.size();a++)
        {
            NodeRes = NodeRes->Add(this->CG->GetNode(NodeidList[a])->Content);
        }
        this->CG->GetNode(this->Nodeid)->Content = NodeRes;
    }

    virtual void Backward()
    {
        std::vector<std::string> &NodeidList = this->CG->GetNode(this->Nodeid)->InputNodeidList;
        std::string ThisDNodeid = this->CG->GetDNodeid(this->Nodeid);
        for(size_t a=0;a<NodeidList.size();a++)
        {
            std::string InputDNodeid = this->CG->GetDNodeid(NodeidList[a]);
            this->CG->RegisterOpsAddEdge(InputDNodeid, ThisDNodeid);
            //using r = decltype(this->CG->GetNode(this->Nodeid));//就这行，神秘报错
            auto &InputDNodeOpsParams = this->CG->Opss[InputDNodeid]->Params;
            auto &InputDNodeOpsParamsAddWeight = *(InputDNodeOpsParams.template Get<AddWeightTypePtr>("AddWeight"));
            auto &OutDNodeOpsParamsAddWeight = *(this->Params.template Get<AddWeightTypePtr>("AddWeight"));
            InputDNodeOpsParamsAddWeight[ThisDNodeid] = OutDNodeOpsParamsAddWeight[NodeidList[a]];
        }
    }

    /**在该函数定义Params中的参数.*/
    virtual void ParamsDefinition()
    {
        /**每个输入样本的常数权重.*/
        this->Params.Set("AddWeight", std::make_shared<AddWeightType>());
    }

};