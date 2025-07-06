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
    auto ContentRes = Tensor::SampleMultivariateStandardGaussian(Dim, ShapeVec, Seed, DeviceNum);
    using ContentType = std::remove_pointer_t<std::decay_t<decltype(ContentRes)>>;
    auto ContentPtr = std::shared_ptr<ContentType>(ContentRes);
    return DynamicTensor(ContentPtr);
}