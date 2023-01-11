#pragma once
#include "BaseOptimizer.h"

struct SGDOptimizer:BaseOptimizer
{
public:
    SGDOptimizer(){}
    /**通过学习率初始化自身*/
    SGDOptimizer(float lr)
    {
        OptParams.Set("LearningRate", HyperparameterTypeConst::FLOAT, std::vector<float>{lr});
    }

    virtual void UpdateParams()
    {
        
    }
};