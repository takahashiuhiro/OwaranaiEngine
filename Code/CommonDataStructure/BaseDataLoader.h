#pragma once
#include "HyperElement.h"
#include "../DynamicAutomaticDifferentiation/DynamicTensor.h"

struct BaseDataloader
{
    BaseDataloader(){};

    he DataLoaderParams;

    /**是否用GPU. */
    int DeviceNum = 0;
    int RequiresGrad = 0;

    /**用来分batch的. */
    std::vector<int>BatchList;
    int BatchIndex=0;
    int BatchSize = 1;
    
    void Init(he InputParams);

    void InitCommon();

    /**按顺序拿到所有数据. */
    std::vector<size_t> GetAllIndex();

    /**设置batch size.
     * @param
     * InputBatchSize 新的batchsize
     * IsReset 是不是要重置进度
     */
    void SetBatchSize(int InputBatchSize = 1, bool IsReset = true);

    /**按照预先分好的batch拿数据，每次拿到一个batch. */
    std::vector<size_t> GetBatchIndex();

    std::pair<DynamicTensor,DynamicTensor> GetAllData();
    std::pair<DynamicTensor,DynamicTensor> GetBatchData();

    /**需要如何初始化. */
    virtual void InitDetail() = 0;
    /**需要知道一共有多少数据. */
    virtual size_t GetDataAllNum() = 0;
    /**将数据和输出转化为张量. */
    virtual std::pair<DynamicTensor,DynamicTensor> ChangeDataToTensor(std::vector<size_t>DataIndexs) = 0;

};
