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
    //todo::把inputlist里的节点的output都改成自己的output，用dfs
    for (size_t a = 0; a < OutNodeList.size(); a++)
    {
        //todo::通过dfs查询output节点里用过自己的位置修改为一个dynamicops
    }
    if(TensorPointer!=nullptr)
    {
        TensorPointer.reset();
    }
}

void DynamicTensor::PrintData()
{
    TensorPointer->PrintData();
}

void DynamicTensor::Set(DynamicTensor* ThisTensor, const DynamicTensor* OtherTensor)
{
    ThisTensor->TensorPointer = OtherTensor->TensorPointer;
    ThisTensor->Ops = OtherTensor->Ops;
}

DynamicTensor DynamicTensor::Add(DynamicTensor& InputFirst, DynamicTensor& InputSecond, bool RequiresGrad)
{
    DynamicTensor Res = DynamicTensor(std::shared_ptr<Tensor>(InputFirst.TensorPointer->Add(InputSecond.TensorPointer.get())));
    if (!RequiresGrad)return Res;
    Res.Ops.DynamicOpsType = OpsType::Add;
    Res.Ops.InputOpsList = {DynamicOps(&InputFirst), DynamicOps(&InputSecond)};
    InputFirst.OutNodeList.push_back(&Res);
    InputSecond.OutNodeList.push_back(&Res);
    return Res;
}
