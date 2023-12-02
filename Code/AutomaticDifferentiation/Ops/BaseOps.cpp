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
    /**每个输入样本选择的维度.*/
    this->Params.Set("SelectDim", std::make_shared<SelectDimType>());
    this->Params.Set("SelectDims", std::make_shared<SelectDimsType>());
    /**每个输入样本广播的shape对.*/
    this->Params.Set("BroadCastTo", std::make_shared<BroadCastToType>());
}

void BaseOps::Backward()
{
    Log::Assert(0, std::string("This Node's Backward Was Node Implemented: ")+Nodeid);
}

void BaseOps::CGForwardProcess()
{
    CGForwardProcessDropout();
}

void BaseOps::CGForwardProcessDropout()
{
    if(!CG->CGMode)return;
    if(!CG->GetNode(Nodeid)->Property.Get<bool>("Dropout"))return;
    float P = 1.- CG->GetNode(Nodeid)->Property.Get<float>("DropoutP");
    Tensor* DropoutTensor = CG->GetNode(Nodeid)->GetContent()->Copy();
    DropoutTensor->FillRandomValBernoulli(P);
    Tensor* DropoutTensorDotP = DropoutTensor->MulScalar(1/P);
    delete DropoutTensor;
    CG->GetNode(Nodeid)->AssignContent(DropoutTensorDotP->EleMul(CG->GetNode(Nodeid)->GetContent()));
    delete DropoutTensorDotP;
}

void BaseOps::ForwardProcess()
{
    Forward();
    CGForwardProcess();
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

void BaseOps::SetSelectDim(SelectDimType InputNodeSelectDim)
{
    auto &OutDNodeOpsParamsSelectDim = *(Params.template Get<SelectDimTypePtr>("SelectDim"));
    for(auto& CGNodeidItem:InputNodeSelectDim)
    {
        OutDNodeOpsParamsSelectDim[CGNodeidItem.first] = CGNodeidItem.second;
    }
}

size_t BaseOps::GetSelectDim(std::string InputNodeid)
{
    auto &OutDNodeOpsParamsSelectDim = *(Params.template Get<SelectDimTypePtr>("SelectDim"));
    Log::Assert(OutDNodeOpsParamsSelectDim.find(InputNodeid)!=OutDNodeOpsParamsSelectDim.end(), std::string("This Node Is Not Set SelectDim Node id:")+InputNodeid);
    return OutDNodeOpsParamsSelectDim[InputNodeid];
}

void BaseOps::SetBroadCastTo(BroadCastToType BroadCastToShape)
{
    auto &OutDNodeOpsParamsBroadCastTo = *(Params.template Get<BroadCastToTypePtr>("BroadCastTo"));
    for(auto& CGNodeidItem:BroadCastToShape)
    {
        OutDNodeOpsParamsBroadCastTo[CGNodeidItem.first] = CGNodeidItem.second;
    }
}

std::vector<size_t> BaseOps::GetBroadCastTo(std::string InputNodeid)
{
    auto &OutDNodeOpsParamsBroadCastTo = *(Params.template Get<BroadCastToTypePtr>("BroadCastTo"));
    Log::Assert(OutDNodeOpsParamsBroadCastTo.find(InputNodeid)!=OutDNodeOpsParamsBroadCastTo.end(), std::string("This Node Is Not Set BroadCastTo Node id:")+InputNodeid);
    return OutDNodeOpsParamsBroadCastTo[InputNodeid];
}

void BaseOps::SetSelectDims(SelectDimsType InputNodeSelectDims)
{
    auto &OutDNodeOpsParamsSelectDims = *(Params.template Get<SelectDimsTypePtr>("SelectDims"));
    for(auto& CGNodeidItem:InputNodeSelectDims)
    {
        OutDNodeOpsParamsSelectDims[CGNodeidItem.first] = CGNodeidItem.second;
    }
}

std::vector<size_t> BaseOps::GetSelectDims(std::string InputNodeid)
{
    auto &OutDNodeOpsParamsSelectDims = *(Params.template Get<SelectDimsTypePtr>("SelectDims"));
    Log::Assert(OutDNodeOpsParamsSelectDims.find(InputNodeid)!=OutDNodeOpsParamsSelectDims.end(), std::string("This Node Is Not Set SelectDims Node id:")+InputNodeid);
    return OutDNodeOpsParamsSelectDims[InputNodeid];
}

void BaseOps::SetEleExponent(float Exponent)
{
    Params.Set("EleExponent", Exponent);
}

float BaseOps::GetEleExponent()
{
    return Params.Get<float>("EleExponent");
}

void BaseOps::SetEleBaseNum(float BaseNum)
{
    Params.Set("EleBaseNum", BaseNum);
}

float BaseOps::GetEleBaseNum()
{
    return Params.Get<float>("EleBaseNum");
}

void BaseOps::SetSelectDimSingle(int SelectDimIndex)
{
    Params.Set("SelectDimIndex", SelectDimIndex);
}

int BaseOps::GetSelectDimSingle()
{
    return Params.Get<int>("SelectDimIndex");
}