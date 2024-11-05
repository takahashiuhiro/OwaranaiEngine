#pragma once
#include "HyperElement.h"
#include "../DynamicAutomaticDifferentiation/DynamicTensor.h"

template<typename LoaderDataType>
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

    
    LoaderDataType GetAllData();
    LoaderDataType GetBatchData();

    /**需要如何初始化. */
    virtual void InitDetail() = 0;
    /**需要知道一共有多少数据. */
    virtual size_t GetDataAllNum() = 0;
    /**将数据和输出转化为张量. */
    virtual LoaderDataType ChangeDataToTensor(std::vector<size_t>DataIndexs) = 0;

};

template<typename LoaderDataType>
void BaseDataloader<LoaderDataType>::Init(he InputParams)
{
    DataLoaderParams = InputParams;
    InitCommon();
    InitDetail();
    SetBatchSize();
}

template<typename LoaderDataType>
void BaseDataloader<LoaderDataType>::InitCommon()
{
    if (DataLoaderParams.In("DeviceNum"))DeviceNum = DataLoaderParams["DeviceNum"].i();
	else DeviceNum = 0;
    if (DataLoaderParams.In("RequiresGrad"))RequiresGrad = DataLoaderParams["RequiresGrad"].i();
	else RequiresGrad = 0;
}

template<typename LoaderDataType>
std::vector<size_t> BaseDataloader<LoaderDataType>::GetAllIndex()
{
    std::vector<size_t>Res;
    size_t DataNum = GetDataAllNum();
    for(size_t a=0;a<DataNum;a++)Res.push_back(a);
    return Res;
}

template<typename LoaderDataType>
void BaseDataloader<LoaderDataType>::SetBatchSize(int InputBatchSize, bool IsReset)
{
    size_t DataNum = GetDataAllNum();
    if(IsReset)
    {
        BatchIndex = 0;
        BatchList = GenerateUniqueRandomNumbers(DataNum, 0, DataNum-1);
    }
    BatchSize = InputBatchSize;
}

template<typename LoaderDataType>
std::vector<size_t> BaseDataloader<LoaderDataType>::GetBatchIndex()
{
    std::vector<size_t>Res;
    int ProtoBatchIndex = BatchIndex;
    for(;;)
    {
        if(Res.size() >= BatchSize || BatchIndex > BatchList.size() - 1)break;
        Res.push_back(BatchList[BatchIndex]);
        BatchIndex ++;
    }
    return Res;
}

template<typename LoaderDataType>
LoaderDataType BaseDataloader<LoaderDataType>::GetAllData()
{
    return ChangeDataToTensor(GetAllIndex());
}

template<typename LoaderDataType>
LoaderDataType BaseDataloader<LoaderDataType>::GetBatchData()
{
    return ChangeDataToTensor(GetBatchIndex());
}