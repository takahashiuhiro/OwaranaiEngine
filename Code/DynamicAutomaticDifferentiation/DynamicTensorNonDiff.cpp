#include "DynamicTensor.h"

DynamicTensor DynamicTensor::Cholesky()
{
    using ContentType = std::decay_t<decltype(*(Ops->TensorPointer))>;
    auto CholeskyRes = std::shared_ptr<ContentType>(Ops->TensorPointer->Cholesky());
    return DynamicTensor(CholeskyRes, Ops->RequiresGrad);
}