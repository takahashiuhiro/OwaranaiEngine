#pragma once
#include "BaseOptimizer.h"

struct SGDOptimizerInputTypeConst
{
    /**通过Tensor进行优化，用于FF类算法*/
    static const int BY_TENSOR = 0;
    /**通过CGNode进行优化，一般用于反向算法*/
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

    /**不同的输入规格代表不同的计算方式，在参数里可以找到对应参数*/
    virtual void UpdateParams()
    {
        float LearningRate = (*(OptParams).Get<std::vector<float>>("LearningRate"))[0];
        LearningRate *= -1;
        if((*OptParams.Get<std::vector<int>>("InputType"))[0] == SGDOptimizerInputTypeConst::BY_CGNODE)
        {
            for(int a = 0;a<InputCGNodeList.size();a++)
            {
                Tensor* Grandient = DerivativeCGNodeList[a]->NodeContent;
                InputCGNodeList[a]->NodeContent = InputCGNodeList[a]->NodeContent->Add(Grandient->MulScalar(LearningRate));
            }
        }
        if((*OptParams.Get<std::vector<int>>("InputType"))[0] == SGDOptimizerInputTypeConst::BY_TENSOR)
        {
            for(int a = 0;a<InputTensorList.size();a++)
            {
                Tensor* Grandient = DerivativeTensorList[a];
                InputTensorList[a] = InputTensorList[a]->Add(Grandient->MulScalar(LearningRate));
            }
        }
    }

    virtual void Clear()
    {
        InputTensorList.clear();
        DerivativeTensorList.clear();
        InputCGNodeList.clear();
        DerivativeCGNodeList.clear();
    }
};