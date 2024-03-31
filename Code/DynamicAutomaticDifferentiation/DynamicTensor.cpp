#include "DynamicTensor.h"

DynamicTensor::DynamicTensor(Tensor* InputTensor)
{
    TensorPointer = InputTensor;
    DynamicTensorInit();
}
std::shared_ptr<DynamicTensor> DynamicTensor::CreateDynamicTensor(Tensor* InputTensor)
{
    std::shared_ptr<DynamicTensor> Res = std::make_shared<DynamicTensor>(InputTensor);
    return Res;
}
std::shared_ptr<DynamicTensor> DynamicTensor::CreateDynamicTensor(std::vector<size_t>shape)
{
    std::shared_ptr<DynamicTensor> Res = std::make_shared<DynamicTensor>(new Tensor(shape));
    return Res;
}
std::shared_ptr<DynamicTensor> DynamicTensor::CreateDynamicTensor(std::vector<size_t>shape, size_t DeviceNum)
{
    std::shared_ptr<DynamicTensor> Res = std::make_shared<DynamicTensor>(new Tensor(shape, DeviceNum));
    return Res;
}
std::shared_ptr<DynamicTensor> DynamicTensor::CreateDynamicTensor(std::vector<size_t>shape, size_t DeviceNum, std::vector<float>InputData)
{
    std::shared_ptr<DynamicTensor> Res = std::make_shared<DynamicTensor>(new Tensor(shape, DeviceNum, InputData));
    return Res;
}

void DynamicTensor::DynamicTensorInit()
{
    DynamicTensorParamsInit();
}

void DynamicTensor::DynamicTensorParamsInit()
{
    Params = he::NewDict();
    Params["requires_grad"] = 0;
}

DynamicTensor::~DynamicTensor()
{
    if(TensorPointer!=nullptr)
    {
        delete TensorPointer;
    }
}

void DynamicTensor::PrintData()
{
    TensorPointer->PrintData();
}

std::shared_ptr<DynamicTensor> DynamicTensor::Add(DynamicTensor* Input1, DynamicTensor* Input2, bool NeedGrad)
{
    std::shared_ptr<DynamicTensor> Res = std::make_shared<DynamicTensor>(Input1->TensorPointer->Add(Input2->TensorPointer));
    Res->OpsType = DynamicTensorOpsType::Add;
    if(NeedGrad)
    {
        bool Input1Flag = Input1->Params["requires_grad"].i();
        bool Input2Flag = Input2->Params["requires_grad"].i();
        if(Input1Flag||Input2Flag)
        {
            Res->Params["requires_grad"] = 1;
            Res->InputList.push_back(Input1);
            Res->InputList.push_back(Input2);
            if(Input1Flag)Input1->OutputList.push_back(Res.get());
            if(Input2Flag)Input2->OutputList.push_back(Res.get());
        }
    }
    return Res;
}
