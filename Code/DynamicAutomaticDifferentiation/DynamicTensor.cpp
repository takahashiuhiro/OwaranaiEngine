#include "DynamicTensor.h"

DynamicOps::DynamicOps(DynamicTensor* DynamicTensorNode){leafNode = DynamicTensorNode;}

DynamicTensor::DynamicTensor(std::shared_ptr<Tensor> InputTensor,bool IsRequiresGrad)
{
    TensorPointer = InputTensor;
    RequiresGrad = IsRequiresGrad;
    if (IsRequiresGrad)
    {
        Grad = std::make_shared<DynamicTensor>(std::shared_ptr<Tensor>(TensorPointer->CopyNewEmptyTensor()),false);
        Grad->TensorPointer->FillArray(0);
    }
}
DynamicTensor::DynamicTensor(const DynamicTensor& Other){Set(this, &Other);}
DynamicTensor DynamicTensor::CreateDynamicTensor(std::shared_ptr<Tensor> InputTensor, bool IsRequiresGrad){return DynamicTensor(InputTensor);}
DynamicTensor DynamicTensor::CreateDynamicTensor(std::vector<size_t>shape, bool IsRequiresGrad, size_t DeviceNum){return DynamicTensor(std::make_shared<Tensor>(shape, DeviceNum));}
DynamicTensor DynamicTensor::CreateDynamicTensor(std::vector<size_t>shape, std::vector<float>InputData, bool IsRequiresGrad, size_t DeviceNum){return DynamicTensor(std::make_shared<Tensor>(shape, DeviceNum, InputData));}
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

void DynamicTensor::SetForwardHistory(DynamicTensor& InputRes, size_t InputOptType, std::vector<DynamicTensor*>OpsList, he InputPrams, bool IsRequiresGrad)
{   
    if (!IsRequiresGrad)return;
    InputRes.Ops.DynamicOpsType = InputOptType;
    InputRes.Ops.InputOpsList = {};
    InputRes.Ops.Params = InputPrams;
    for (size_t a = 0; a < OpsList.size(); a++)
    {
        InputRes.Ops.InputOpsList.push_back(DynamicOps(OpsList[a]));
        OpsList[a]->OutNodeList.insert(&InputRes);
        if (OpsList[a]->RequiresGrad)
        {
            InputRes.RequiresGrad = true;
            InputRes.Grad = std::make_shared<DynamicTensor>(std::shared_ptr<Tensor>(InputRes.TensorPointer->CopyNewEmptyTensor()), false);
            InputRes.Grad->TensorPointer->FillArray(0);
        }
    }
}

void DynamicTensor::Backward(DynamicTensor LossResult)
{

}

void DynamicTensor::Backward()
{
    Grad = std::make_shared<DynamicTensor>(std::shared_ptr<Tensor>(TensorPointer->Copy()));//在这种情况下启动这个函数的时候自己的值就是初始的梯度值
    std::map<DynamicOps*, std::shared_ptr<DynamicTensor>>GradGlobalResult;//计算完毕的全局导数(存在的意义主要是有些节点没有实体，不能直接存到tensor里)
    std::map<DynamicOps*, std::shared_ptr<DynamicTensor>>ForwardGlobalResult;//计算完毕的实体
    std::map<std::pair<DynamicOps*, DynamicOps*>, std::shared_ptr<DynamicTensor>>PartGradGlobalResult;//计算完毕的输入->输出的导数结果(对应一个输入节点多个输出节点的情况)
    GradGlobalResult[&(this->Ops)] = Grad;
    BackwardDfs(GradGlobalResult, ForwardGlobalResult, PartGradGlobalResult, &(this->Ops));
}

void DynamicTensor::BackwardDfs(std::map<DynamicOps*, std::shared_ptr<DynamicTensor>>& GradGlobalResult, std::map<DynamicOps*, std::shared_ptr<DynamicTensor>>& ForwardGlobalResult, std::map<std::pair<DynamicOps*, DynamicOps*>, std::shared_ptr<DynamicTensor>>& PartGradGlobalResult,DynamicOps* CurOps)
{
    //todo
    //1.求实体::如果ops没有实体，那就先dfs(另写一个函数做这个事，结束条件是到一个有实体，或者没有实体但是求过了)求出来实体存在ForwardGlobalResult里
    for (size_t a = 0; a < CurOps->InputOpsList.size(); a++)
    {
    }
    //2.通过当前这一层已有的求出输入节点到该节点的导数，每次算出来一个部分导数，就检查是不是所有部分都算出来了，如果是那就向上，不是就停止
    //3.如果碰到不需要算导数的那就不用求
    //4.无论是否有实体都要在map里存一份，有实体的还要额外在tensor里存
}

void DynamicTensor::Set(DynamicTensor* ThisTensor, const DynamicTensor* OtherTensor)
{
    ThisTensor->TensorPointer = OtherTensor->TensorPointer;
    ThisTensor->Ops = OtherTensor->Ops;
}

DynamicTensor DynamicTensor::Add(std::vector<DynamicTensor*>InputList, he InputPrams, bool IsRequiresGrad)
{
    DynamicTensor Res = DynamicTensor(std::shared_ptr<Tensor>(InputList[0]->TensorPointer->Add(InputList[1]->TensorPointer.get())));
    SetForwardHistory(Res, OpsType::Add, InputList, InputPrams, IsRequiresGrad);
    return Res;
}
