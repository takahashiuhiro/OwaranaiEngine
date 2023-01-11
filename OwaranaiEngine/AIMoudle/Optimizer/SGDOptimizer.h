#pragma once
#include "BaseOptimizer.h"

struct SGDOptimizerInputTypeConst
{
    static const int BY_TENSOR = 0;
    static const int BY_CGNODE = 1;
};

struct SGDOptimizer:BaseOptimizer
{
public:
    SGDOptimizer(){}
    /**通过学习率初始化自身*/
    SGDOptimizer(float LearningRate, int InputType)
    {
        OptParams.Set("LearningRate", HyperparameterTypeConst::FLOAT, std::vector<float>{LearningRate});
        OptParams.Set("InputType", HyperparameterTypeConst::INT, std::vector<int>{InputType});
    }

    /**分别对应不同的输入方式*/
    std::vector<Tensor*>InputTensorList;
    std::vector<Tensor*>DerivativeTensorList;
    std::vector<CGNode*>InputCGNodeList;
    std::vector<CGNode*>DerivativeCGNodeList;

    virtual void UpdateParams()
    {
        /**不同的输入规格代表不同的计算方式*/
        if((*OptParams.Get<std::vector<int>>("InputType"))[0] == SGDOptimizerInputTypeConst::BY_CGNODE)
        {

        }
        if((*OptParams.Get<std::vector<int>>("InputType"))[0] == SGDOptimizerInputTypeConst::BY_TENSOR)
        {
            
        }
    }
};