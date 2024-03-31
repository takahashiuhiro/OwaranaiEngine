#include "DynamicTensor.h"

DynamicTensor::DynamicTensor(Tensor* InputTensor){TensorPointer = InputTensor;}
DynamicTensor DynamicTensor::CreateDynamicTensor(Tensor* InputTensor){return DynamicTensor(InputTensor);}
DynamicTensor DynamicTensor::CreateDynamicTensor(std::vector<size_t>shape){return DynamicTensor(new Tensor(shape));}
DynamicTensor DynamicTensor::CreateDynamicTensor(std::vector<size_t>shape, size_t DeviceNum){return DynamicTensor(new Tensor(shape, DeviceNum));}
DynamicTensor DynamicTensor::CreateDynamicTensor(std::vector<size_t>shape, size_t DeviceNum, std::vector<float>InputData){return DynamicTensor(new Tensor(shape, DeviceNum, InputData));}
DynamicTensor DynamicTensor::CreateVector(std::vector<float>InputData, size_t DeviceNum){return DynamicTensor(new Tensor({1,InputData.size()}, DeviceNum, InputData));}

DynamicTensor::~DynamicTensor()
{
    if(TensorPointer!=nullptr)
    {
        delete TensorPointer;
        TensorPointer=nullptr;
    }
}

DynamicTensor DynamicTensor::operator + (DynamicTensor& Other)
{
    if(TensorPointer->shape.size()==Other.TensorPointer->shape.size())
    {
        bool ShapeCheck = true;
        for(size_t a=0;a<TensorPointer->shape.size();a++)
        {
            ShapeCheck &= TensorPointer->shape[a] == Other.TensorPointer->shape[a];
        }
        if(ShapeCheck)return DynamicTensor(TensorPointer->Add(Other.TensorPointer));
    }
    std::vector<size_t>FinalShapeVec;
    FinalShapeVec.resize(std::max(TensorPointer->shape.size(), Other.TensorPointer->shape.size()));
    for(size_t a =0;a<FinalShapeVec.size();a++)FinalShapeVec[a] = 1;
    for(size_t a =0;a<FinalShapeVec.size();a++)
    {
        int CurShapeVec = FinalShapeVec.size() - 1 - a;
        int ThisShapeVec = TensorPointer->shape.size() - 1 - a;
        int OtherShapeVec = Other.TensorPointer->shape.size() - 1 - a;
        if(ThisShapeVec<0&&OtherShapeVec>=0)FinalShapeVec[CurShapeVec] = Other.TensorPointer->shape[OtherShapeVec];
        else if(ThisShapeVec>=0&&OtherShapeVec<0)FinalShapeVec[CurShapeVec] = TensorPointer->shape[ThisShapeVec];
        else
        {
            if(Other.TensorPointer->shape[OtherShapeVec]==1)FinalShapeVec[CurShapeVec] = TensorPointer->shape[ThisShapeVec];
            else if(TensorPointer->shape[ThisShapeVec]==1)FinalShapeVec[CurShapeVec] = Other.TensorPointer->shape[OtherShapeVec];
            else
            {
                Log::Assert(TensorPointer->shape[ThisShapeVec]==Other.TensorPointer->shape[OtherShapeVec],"DynamicTensor Add shape error");
                FinalShapeVec[CurShapeVec] = TensorPointer->shape[ThisShapeVec];
            }
        }
    }
    Log::Assert(TensorPointer->CanBroadCastTo(FinalShapeVec)&&Other.TensorPointer->CanBroadCastTo(FinalShapeVec),"DynamicTensor Add shape error");
    Tensor* BroadThis = TensorPointer->BroadCastTo(FinalShapeVec);
    Tensor* BroadOther = Other.TensorPointer->BroadCastTo(FinalShapeVec);
    Tensor* Res = BroadThis->Add(BroadOther);
    delete BroadThis;
    delete BroadOther;
    return DynamicTensor(Res);
}

DynamicTensor DynamicTensor::operator + (DynamicTensor&& Other)
{
    if(TensorPointer->shape.size()==Other.TensorPointer->shape.size())
    {
        bool ShapeCheck = true;
        for(size_t a=0;a<TensorPointer->shape.size();a++)
        {
            ShapeCheck &= TensorPointer->shape[a] == Other.TensorPointer->shape[a];
        }
        if(ShapeCheck)return DynamicTensor(TensorPointer->Add(Other.TensorPointer));
    }
    std::vector<size_t>FinalShapeVec;
    FinalShapeVec.resize(std::max(TensorPointer->shape.size(), Other.TensorPointer->shape.size()));
    for(size_t a =0;a<FinalShapeVec.size();a++)FinalShapeVec[a] = 1;
    for(size_t a =0;a<FinalShapeVec.size();a++)
    {
        int CurShapeVec = FinalShapeVec.size() - 1 - a;
        int ThisShapeVec = TensorPointer->shape.size() - 1 - a;
        int OtherShapeVec = Other.TensorPointer->shape.size() - 1 - a;
        if(ThisShapeVec<0&&OtherShapeVec>=0)FinalShapeVec[CurShapeVec] = Other.TensorPointer->shape[OtherShapeVec];
        else if(ThisShapeVec>=0&&OtherShapeVec<0)FinalShapeVec[CurShapeVec] = TensorPointer->shape[ThisShapeVec];
        else
        {
            if(Other.TensorPointer->shape[OtherShapeVec]==1)FinalShapeVec[CurShapeVec] = TensorPointer->shape[ThisShapeVec];
            else if(TensorPointer->shape[ThisShapeVec]==1)FinalShapeVec[CurShapeVec] = Other.TensorPointer->shape[OtherShapeVec];
            else
            {
                Log::Assert(TensorPointer->shape[ThisShapeVec]==Other.TensorPointer->shape[OtherShapeVec],"DynamicTensor Add shape error");
                FinalShapeVec[CurShapeVec] = TensorPointer->shape[ThisShapeVec];
            }
        }
    }
    Log::Assert(TensorPointer->CanBroadCastTo(FinalShapeVec)&&Other.TensorPointer->CanBroadCastTo(FinalShapeVec),"DynamicTensor Add shape error");
    Tensor* BroadThis = TensorPointer->BroadCastTo(FinalShapeVec);
    Tensor* BroadOther = Other.TensorPointer->BroadCastTo(FinalShapeVec);
    Tensor* Res = BroadThis->Add(BroadOther);
    delete BroadThis;
    delete BroadOther;
    return DynamicTensor(Res);
}

void DynamicTensor::PrintData()
{
    TensorPointer->PrintData();
}

