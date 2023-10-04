#include "BaseLayer.h"

bool BaseLayer::IsRootNode()
{
    return ParentLayer == nullptr;
}

void BaseLayer::CommonInit(BaseLayer* InputParentLayer, std::string InputLayerName, size_t ThisDeviceNum)
{
    ParentLayer = InputParentLayer;
    LayerName = InputLayerName;
    DeviceNum = ThisDeviceNum;
    if(IsRootNode())
    {
        CG = std::make_shared<ComputationalGraph>();
        this->PreName = "";//InputLayerName;
    }
    else
    {
        CG = InputParentLayer->CG;
        this->PreName = GetLinkName(ParentLayer->PreName, ParentLayer->LayerName);
    }
}

void BaseLayer::RegisterLayer(std::shared_ptr<BaseLayer>InputLayer)
{
    SubLayers[InputLayer->LayerName] = InputLayer;
}

void BaseLayer::RegisterWeightNode(std::string InputNodeid,std::vector<size_t>InputTensorShape)
{
    CG->RegisterWeightNode(GetLayerNodeName(InputNodeid), InputTensorShape);
    WeightNodeArray.push_back(InputNodeid);
}

std::string BaseLayer::GetLayerNodeName(std::string InputNodeName)
{
    return GetLinkName(GetLinkName(PreName, LayerName), InputNodeName);
}

void BaseLayer::SaveToFile(std::string SavePath)
{
    std::ofstream OpenedFile(SavePath, std::ios::binary);
    Log::Assert(OpenedFile.is_open(), std::string("This File Is Not Opened::") + SavePath);
    std::vector<std::string> NodeList =  this->GetAllSubLayersNodeDfs();
    size_t NodeListSize = NodeList.size();
    OpenedFile.write(reinterpret_cast<const char*>(&NodeListSize), sizeof(NodeListSize));
    for(size_t a = 0;a<NodeListSize;a++)
    {
        SaveToFileString(OpenedFile, NodeList[a]);
        CG->GetNode(GetLayerNodeName(NodeList[a]))->GetContent()->SaveToFile(OpenedFile);
    }
    OpenedFile.close();
}

void BaseLayer::LoadFromFile(std::string LoadPath)
{
    std::ifstream OpenedFile(LoadPath, std::ios::binary);
    Log::Assert(OpenedFile.is_open(), std::string("This File Is Not Opened::") + LoadPath);
    size_t NodeListSize;
    OpenedFile.read(reinterpret_cast<char*>(&NodeListSize), sizeof(NodeListSize));
    for(size_t a = 0;a<NodeListSize;a++)
    {
        std::string ThisNodeName = LoadFromFileString(OpenedFile);
        Tensor* ThisNodeContent = Tensor::CreateTensorByLoadPath(OpenedFile, DeviceNum);
        std::cout<<"????: "<<ThisNodeName<<std::endl;
        CG->GetNode(GetLayerNodeName(ThisNodeName))->AssignContent(ThisNodeContent);
    }
}

std::vector<std::string> BaseLayer::GetAllSubLayersNodeDfs()
{
    std::vector<std::string> ReturnNodeIdList = WeightNodeArray;
    for(auto &a:SubLayers)
    {
        std::vector<std::string>TMPList = a.second->GetAllSubLayersNodeDfs(false);
        for(size_t b=0; b<TMPList.size();b++)
        {
            ReturnNodeIdList.push_back(TMPList[b]);
        }
    }
    return ReturnNodeIdList;
}

std::vector<std::string> BaseLayer::GetAllSubLayersNodeDfs(bool AutoFlag)
{
    std::vector<std::string> ReturnNodeIdList = WeightNodeArray;
    for(auto &a:SubLayers)
    {
        std::vector<std::string>TMPList = a.second->GetAllSubLayersNodeDfs(false);
        for(size_t b=0; b<TMPList.size();b++)
        {
            ReturnNodeIdList.push_back(TMPList[b]);
        }
    }
    if(!AutoFlag)
    {
        for(size_t a = 0;a<ReturnNodeIdList.size();a++)
        {
            ReturnNodeIdList[a] = GetLinkName(LayerName, ReturnNodeIdList[a]);
        }
    }
    return ReturnNodeIdList;
}

std::string BaseLayer::GetLinkName(std::string PreStr, std::string NxtStr)
{
    if(PreStr == std::string(""))return NxtStr;
    return PreStr + std::string(".") + NxtStr;
}