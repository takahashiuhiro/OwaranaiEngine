#include "BaseDataLoader.h"

void BaseDataloader::Init(he InputParams)
{
    DataLoaderParams = InputParams;
    InitCommon();
    InitDetail();
    SetBatchSize();
}

void BaseDataloader::InitCommon()
{
    if (DataLoaderParams.In("DeviceNum"))DeviceNum = DataLoaderParams["DeviceNum"].i();
	else DeviceNum = 0;
    if (DataLoaderParams.In("RequiresGrad"))RequiresGrad = DataLoaderParams["RequiresGrad"].i();
	else RequiresGrad = 0;
}

std::vector<size_t> BaseDataloader::GetAllIndex()
{
    std::vector<size_t>Res;
    size_t DataNum = GetDataAllNum();
    for(size_t a=0;a<DataNum;a++)Res.push_back(a);
    return Res;
}

void BaseDataloader::SetBatchSize(int InputBatchSize, bool IsReset)
{
    size_t DataNum = GetDataAllNum();
    if(IsReset)
    {
        BatchIndex = 0;
        BatchList = GenerateUniqueRandomNumbers(DataNum+1, 0, DataNum);
    }
    BatchSize = InputBatchSize;
}

std::vector<size_t> BaseDataloader::GetBatchIndex()
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

std::pair<DynamicTensor,DynamicTensor> BaseDataloader::GetAllData()
{
    return ChangeDataToTensor(GetAllIndex());
}

std::pair<DynamicTensor,DynamicTensor> BaseDataloader::GetBatchData()
{
    return ChangeDataToTensor(GetBatchIndex());
}