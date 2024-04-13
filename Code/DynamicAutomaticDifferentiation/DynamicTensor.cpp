#include "DynamicTensor.h"

DynamicOps::DynamicOps(DynamicTensor* DynamicTensorNode){leafNode = DynamicTensorNode;}

DynamicTensor::DynamicTensor(std::shared_ptr<Tensor> InputTensor){TensorPointer = InputTensor;}
DynamicTensor::DynamicTensor(const DynamicTensor& Other){Set(this, &Other);}
DynamicTensor DynamicTensor::CreateDynamicTensor(std::shared_ptr<Tensor> InputTensor){return DynamicTensor(InputTensor);}
DynamicTensor DynamicTensor::CreateDynamicTensor(std::vector<size_t>shape, size_t DeviceNum){return DynamicTensor(std::make_shared<Tensor>(shape, DeviceNum));}
DynamicTensor DynamicTensor::CreateDynamicTensor(std::vector<size_t>shape, std::vector<float>InputData, size_t DeviceNum){return DynamicTensor(std::make_shared<Tensor>(shape, DeviceNum, InputData));}
DynamicTensor DynamicTensor::CreateVector(std::vector<float>InputData, size_t DeviceNum)
{
    std::vector<size_t>ThisNewShape = { 1U,InputData.size() };
    return DynamicTensor(std::make_shared<Tensor>(ThisNewShape, DeviceNum, InputData));
}

DynamicTensor::~DynamicTensor()
{
    SetOutputList(Ops,this);
    for (auto& element : OutNodeList)SetInputList(element->Ops, this);
    if(TensorPointer!=nullptr)
    {
        TensorPointer.reset();
    }
}

void DynamicTensor::SetOutputList(DynamicOps& CurOps, DynamicTensor* TargetOutputNode)
{
    for (size_t a = 0; a < CurOps.InputOpsList.size(); a++)
    {
        if (CurOps.InputOpsList[a].leafNode != nullptr)
        {
            for (auto& element : TargetOutputNode->OutNodeList)CurOps.InputOpsList[a].leafNode->OutNodeList.insert(element);
            CurOps.InputOpsList[a].leafNode->OutNodeList.erase(TargetOutputNode);
            continue;
        }
        SetOutputList(CurOps.InputOpsList[a], TargetOutputNode);
    }
}

void DynamicTensor::SetInputList(DynamicOps& CurOps, DynamicTensor* TargetOutputNode)
{
    for (size_t a = 0; a < CurOps.InputOpsList.size(); a++)
    {
        if (CurOps.InputOpsList[a].leafNode !=nullptr)
        {
            if (CurOps.InputOpsList[a].leafNode == TargetOutputNode)
            {
                CurOps.InputOpsList[a] = TargetOutputNode->Ops;
                CurOps.InputOpsList[a].leafNode = nullptr;
            }
        }
        else SetInputList(CurOps.InputOpsList[a], TargetOutputNode);
    }
}

void DynamicTensor::PrintData()
{
    TensorPointer->PrintData();
}

void DynamicTensor::SetForwardHistory(DynamicTensor& InputRes, size_t InputOptType, std::vector<DynamicTensor*>OpsList, bool IsRequiresGrad)
{   
    if (!IsRequiresGrad)return;
    InputRes.Ops.DynamicOpsType = InputOptType;
    InputRes.Ops.InputOpsList = {};
    for (size_t a = 0; a < OpsList.size(); a++)
    {
        InputRes.Ops.InputOpsList.push_back(DynamicOps(OpsList[a]));
        OpsList[a]->OutNodeList.insert(&InputRes);
    }
}

void DynamicTensor::Set(DynamicTensor* ThisTensor, const DynamicTensor* OtherTensor)
{
    ThisTensor->TensorPointer = OtherTensor->TensorPointer;
    ThisTensor->Ops = OtherTensor->Ops;
}

DynamicTensor DynamicTensor::Add(DynamicTensor& InputFirst, DynamicTensor& InputSecond, bool IsRequiresGrad)
{
    DynamicTensor Res = DynamicTensor(std::shared_ptr<Tensor>(InputFirst.TensorPointer->Add(InputSecond.TensorPointer.get())));
    SetForwardHistory(Res, OpsType::Add, { &InputFirst ,&InputSecond }, IsRequiresGrad);
    return Res;
}
