#include "DynamicTensor.h"

DynamicTensor DynamicTensor::Cholesky()
{
    using ContentType = std::decay_t<decltype(*(Ops->TensorPointer))>;
    auto CholeskyRes = std::shared_ptr<ContentType>(Ops->TensorPointer->Cholesky());
    return DynamicTensor(CholeskyRes, Ops->RequiresGrad);
}

DynamicTensor DynamicTensor::SampleFromStdGaussian(int Dim, std::vector<int> InputVec, int Seed,int DeviceNum)
{
    std::vector<size_t> ShapeVec;
    for(auto&it:InputVec)ShapeVec.push_back(it);
    if(Seed == -1)Seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto ContentRes = Tensor::SampleMultivariateStandardGaussian(Dim, ShapeVec, Seed, DeviceNum);
    using ContentType = std::remove_pointer_t<std::decay_t<decltype(ContentRes)>>;
    auto ContentPtr = std::shared_ptr<ContentType>(ContentRes);
    return DynamicTensor(ContentPtr);
}

DynamicTensor DynamicTensor::SampleFromOtherGaussian(int Dim, std::vector<int> InputVec, DynamicTensor Mean, DynamicTensor Var,DynamicTensor VarL, int Seed,int DeviceNum)
{
    if(Seed == -1)Seed = std::chrono::system_clock::now().time_since_epoch().count();
    DynamicTensor STDSamples = DynamicTensor::SampleFromStdGaussian(Dim, InputVec, Seed, DeviceNum);
    if(VarL.Ops == nullptr)VarL= Var.Cholesky();
    InputVec.push_back(Dim);
    auto OutputVec = InputVec;
    InputVec.push_back(1);
    return Mean + (VarL%STDSamples.View(InputVec)).View(OutputVec);
}