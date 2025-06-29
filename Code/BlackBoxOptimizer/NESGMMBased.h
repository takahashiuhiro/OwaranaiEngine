#pragma once
#include "BaseBlackBoxOptimizer.h"

// 标准高斯分布
struct GaussianDistribution
{
    DynamicTensor Mean;
    DynamicTensor Var;
};

struct GMM
{
    DynamicTensor Mean;
    DynamicTensor Var;
    DynamicTensor PartialRate;
};



template<typename TargetType>
struct NESGMMBased: public BaseBlackBoxOptimizer<TargetType>
{

    int DimNum; //优化变量维度
    int SampleNum; //采样数
    int CosmosNum; //粒子数
    double LearingRate_Var; //学习率
    double LearingRate_Mean; //学习率



    virtual void ParamsInit()
    {
        DimNum = this->Params["DimNum"].i();
        SampleNum = this->Params.DictGet("SampleNum", 30).i();
        CosmosNum = this->Params.DictGet("CosmosNum", 30).i();
        LearingRate_Mean = this->Params.DictGet("LearingRate_Mean", 0.1).f();
        LearingRate_Var = this->Params.DictGet("LearingRate_Var", 0.02).f();
    }

    virtual DynamicTensor Solve()
    {        
        // 暂时返回值
        return DynamicTensor();
    }
    
};
