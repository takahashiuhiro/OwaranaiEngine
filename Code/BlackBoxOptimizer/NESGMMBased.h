#pragma once
#include "BaseBlackBoxOptimizer.h"

struct GMM
{
    DynamicTensor Mean;// 均值
    DynamicTensor Var, VarL; // 协方差，协方差的LU分解
    DynamicTensor PartialRate;// 不同高斯的系数

    void Init(DynamicTensor InputMean, DynamicTensor InputVar, DynamicTensor InputPartialRate)
    {
        Mean = InputMean;
        Var = InputVar;
        PartialRate = InputPartialRate;
        VarL = Var.Cholesky();
    }
};


template<typename TargetType>
struct NESGMMBased: public BaseBlackBoxOptimizer<TargetType>
{

    int DimNum; //优化变量维度
    int SampleNum; //采样数
    int CosmosNum; //GMM的高斯数
    double LearingRate_Var; //学习率
    double LearingRate_Mean; //学习率

    GMM TargetDistribution; // 目标分布

    bool IsWarmStart = false; // 是不是已经热启动了，如果热启动了就可以跳过随机初始化

    virtual void ParamsInit()
    {
        DimNum = this->Params["DimNum"].i();
        SampleNum = this->Params.DictGet("SampleNum", 30).i();
        CosmosNum = this->Params.DictGet("CosmosNum", 1).i();
        LearingRate_Mean = this->Params.DictGet("LearingRate_Mean", 0.1).f();
        LearingRate_Var = this->Params.DictGet("LearingRate_Var", 0.02).f();
    }

    /**
     * 初始化目标分布，一般来说都应该给出warm start，如果没给的话就从标准正态分布开始了，这种情况跑得慢是一定的
     */
    void DistributionInit()
    {
        if(IsWarmStart)return;
        DynamicTensor Mean = DynamicTensor({CosmosNum, DimNum}, false, this->DeviceNum);
        Mean.Fill(0);
        DynamicTensor Var = DynamicTensor::CreateUnitTensor({CosmosNum, DimNum, DimNum}, false, this->DeviceNum);
        DynamicTensor PartialRate = DynamicTensor({CosmosNum}, false, this->DeviceNum);
        PartialRate.Fill(1./CosmosNum);
        TargetDistribution.Init(Mean, Var, PartialRate);
    }

    /**
     * 对目标分布进行采样
     */
    DynamicTensor SampleFromTargetDistribution()
    {
        DynamicTensor NewSample = DynamicTensor::SampleFromOtherGaussian(DimNum, {SampleNum}, TargetDistribution.Mean, TargetDistribution.Var, TargetDistribution.VarL, -1, this->DeviceNum);
        return NewSample;
    }

    virtual DynamicTensor Solve()
    {        
        DistributionInit();
        DynamicTensor NewSample = SampleFromTargetDistribution();
        print(NewSample);

        return DynamicTensor(); //暂时返回
    }
    
};
