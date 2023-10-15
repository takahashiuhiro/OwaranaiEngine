#include "BaseOptimizer.h"

void BaseOptimizer::Init()
{
    SetDefaultParams();
}

void BaseOptimizer::Init(std::shared_ptr<ComputationalGraph> InputCG)
{
    SetDefaultParams();
    InitByCG(InputCG);
}

void BaseOptimizer::InitByCG(std::shared_ptr<ComputationalGraph> InputCG)
{
    CG = InputCG;
    RegisterAllTensorsByCG();
}

std::vector<std::string> BaseOptimizer::GetWeightUpdateNodesByCG()
{
    return CG->GetNodesByProperty({"Weight","RequireGrad"},{"Freeze"});
}

void BaseOptimizer::RegisterTensor(std::string TensorName)
{
    TensorMap[TensorName] = std::make_pair(nullptr, nullptr);
    InitTensorConfig(TensorName);
}

void BaseOptimizer::RegisterAllTensorsByCG()
{
    std::vector<std::string> AllTensorNameByCG = GetWeightUpdateNodesByCG();
    for(size_t a =0;a<AllTensorNameByCG.size();a++)
    {
        RegisterTensor(AllTensorNameByCG[a]);
    }
}

void BaseOptimizer::AssignTensor(std::string TensorName,Tensor* ProtoTensor, Tensor* DTensor)
{
    TensorMap[TensorName] = std::make_pair(ProtoTensor, DTensor);
    ProcessAssignTensor(TensorName, ProtoTensor, DTensor);
}

void BaseOptimizer::SyncTensorByCG()
{
    for(auto& TensorPair:TensorMap)
    {
        AssignTensor(TensorPair.first, CG->GetNode(TensorPair.first)->GetContent(),CG->GetNode(CG->GetDNodeid(TensorPair.first))->GetContent());
    }
}

void BaseOptimizer::ClearData()
{
    ResTensorMap.clear();
}

void BaseOptimizer::SyncTensorToCG()
{
    for(auto& TensorPair:ResTensorMap)
    {
        CG->GetNode(TensorPair.first)->AssignContent(TensorPair.second);
    }
}