#pragma once
#include "../DynamicAutomaticDifferentiation/DynamicTensor.h"

template<typename TargetType>
struct BaseBlackBoxOptimizer
{
    he Params;
    int DeviceNum = 0; //计算设备

    TargetType* TargetObj = nullptr; // 优化目标

    void CommonInit(he InputParams) //
    {
        Params = InputParams;
        DeviceNum = Params.DictGet("DeviceNum", 0).i();
    }

    virtual void ParamsInit() = 0; //优化器参数初始化

    void Init(he InputParams) //初始化优化器
    {
        CommonInit(InputParams);
        ParamsInit();
    }

    virtual DynamicTensor Solve() = 0;
};
