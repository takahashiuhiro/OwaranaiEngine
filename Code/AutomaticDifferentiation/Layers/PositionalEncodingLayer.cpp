#include "PositionalEncodingLayer.h"

PositionalEncodingLayer::PositionalEncodingLayer(BaseLayer* ParentThis, std::string ThisLayerName, size_t ThisDeviceNum, size_t DModel, size_t MaxLen)
{
    this->CommonInit(ParentThis,ThisLayerName,ThisDeviceNum);
    this->DModel = DModel;
    this->MaxLen = MaxLen;
    this->RegisterConstNode("PositionalEncoding", {MaxLen, 1U, DModel});
    CG->GetNode(GetLayerNodeName("PositionalEncoding"))->AssignContent(Tensor::PositionalEncoding(DModel, MaxLen, ThisDeviceNum));
}

std::vector<std::string> PositionalEncodingLayer::Forward(std::vector<std::string>InputNodeArray)
{
    std::vector<std::string>ReturnNodeList;
    for(size_t a = 0;a<InputNodeArray.size();a++)
    {
        ReturnNodeList.push_back(OEAutoDiff::Add(CG,{{InputNodeArray[a],1.},{GetLayerNodeName("PositionalEncoding"),1.}}));
    }
    return ReturnNodeList;
}