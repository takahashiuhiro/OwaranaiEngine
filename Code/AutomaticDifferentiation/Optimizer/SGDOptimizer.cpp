#include "SGDOptimizer.h"

void SGDOptimizer::UpdateContent()
{
    double lr = Params.Get<double>("lr");
    for(auto& TensorPair:TensorMap)
    {
        Tensor* MinusTensor = TensorPair.second.second->MulScalar(-lr);
        ResTensorMap[TensorPair.first] = TensorPair.second.first->Add(MinusTensor);
        delete MinusTensor;
    }
}

void SGDOptimizer::SetDefaultParams()
{
    Params.Set("lr", 0.001);
}