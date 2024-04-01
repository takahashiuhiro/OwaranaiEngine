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


DynamicTensor DynamicTensor::operator + (DynamicTensor& Other){return DynamicTensor(OperatorPlus(Other.TensorPointer,0));}
DynamicTensor DynamicTensor::operator + (DynamicTensor&& Other){return DynamicTensor(OperatorPlus(Other.TensorPointer,0));}
DynamicTensor DynamicTensor::operator * (DynamicTensor& Other){return DynamicTensor(OperatorPlus(Other.TensorPointer,1));}
DynamicTensor DynamicTensor::operator * (DynamicTensor&& Other){return DynamicTensor(OperatorPlus(Other.TensorPointer,1));}

DynamicTensor DynamicTensor::Matmul(DynamicTensor& Other){return DynamicTensor(TensorPointer->Matmul(Other.TensorPointer));}
DynamicTensor DynamicTensor::Matmul(DynamicTensor&& Other){return DynamicTensor(TensorPointer->Matmul(Other.TensorPointer));}
DynamicTensor DynamicTensor::T(){return DynamicTensor(TensorPointer->T());}

void DynamicTensor::PrintData()
{
    TensorPointer->PrintData();
}

Tensor* DynamicTensor::OperatorPlus(Tensor*OtherDynamicTensor, size_t InputFunType)
{
    if(TensorPointer->shape.size()==OtherDynamicTensor->shape.size())
    {
        bool ShapeCheck = true;
        for(size_t a=0;a<TensorPointer->shape.size();a++)
        {
            ShapeCheck &= TensorPointer->shape[a] == OtherDynamicTensor->shape[a];
        }
        if(ShapeCheck)
        {
            if(InputFunType==0)return TensorPointer->Add(OtherDynamicTensor);
            if(InputFunType==1)return TensorPointer->EleMul(OtherDynamicTensor);
        }
    }
    std::vector<size_t>FinalShapeVec;
    FinalShapeVec.resize(std::max(TensorPointer->shape.size(), OtherDynamicTensor->shape.size()));
    for(size_t a =0;a<FinalShapeVec.size();a++)FinalShapeVec[a] = 1;
    for(size_t a =0;a<FinalShapeVec.size();a++)
    {
        int CurShapeVec = FinalShapeVec.size() - 1 - a;
        int ThisShapeVec = TensorPointer->shape.size() - 1 - a;
        int OtherShapeVec = OtherDynamicTensor->shape.size() - 1 - a;
        if(ThisShapeVec<0&&OtherShapeVec>=0)FinalShapeVec[CurShapeVec] = OtherDynamicTensor->shape[OtherShapeVec];
        else if(ThisShapeVec>=0&&OtherShapeVec<0)FinalShapeVec[CurShapeVec] = TensorPointer->shape[ThisShapeVec];
        else
        {
            if(OtherDynamicTensor->shape[OtherShapeVec]==1)FinalShapeVec[CurShapeVec] = TensorPointer->shape[ThisShapeVec];
            else if(TensorPointer->shape[ThisShapeVec]==1)FinalShapeVec[CurShapeVec] = OtherDynamicTensor->shape[OtherShapeVec];
            else
            {
                Log::Assert(TensorPointer->shape[ThisShapeVec]==OtherDynamicTensor->shape[OtherShapeVec],"DynamicTensor Add shape error");
                FinalShapeVec[CurShapeVec] = TensorPointer->shape[ThisShapeVec];
            }
        }
    }
    Log::Assert(TensorPointer->CanBroadCastTo(FinalShapeVec)&&OtherDynamicTensor->CanBroadCastTo(FinalShapeVec),"DynamicTensor Add shape error");
    Tensor* BroadThis = TensorPointer->BroadCastTo(FinalShapeVec);
    Tensor* BroadOther = OtherDynamicTensor->BroadCastTo(FinalShapeVec);
    Tensor* Res;
    if(InputFunType==0)Res = BroadThis->Add(BroadOther);
    if(InputFunType==1)Res = BroadThis->EleMul(BroadOther);
    delete BroadThis;
    delete BroadOther;
    return Res;
}