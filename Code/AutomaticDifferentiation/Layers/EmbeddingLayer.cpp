#include "EmbeddingLayer.h"
#include "LinearLayer.h"

EmbeddingLayer::EmbeddingLayer(BaseLayer* ParentThis,std::string ThisLayerName, size_t ThisDeviceNum, size_t NumEmbeddings, size_t EmbeddingDim, std::pair<bool, size_t> PaddingIdx, bool Freeze,std::pair<bool, float> MaxNorm, float NormType, bool ScaleGradByFreq, bool Sparse)
{
    this->CommonInit(ParentThis,ThisLayerName,ThisDeviceNum);
    this->NumEmbeddings = NumEmbeddings;
    this->EmbeddingDim = EmbeddingDim;
    this->PaddingIdx = PaddingIdx;
    this->Freeze = Freeze;
    this->MaxNorm = MaxNorm;
    this->NormType = NormType;
    this->ScaleGradByFreq = ScaleGradByFreq;
    this->Sparse = Sparse;
    this->RegisterLayer(std::make_shared<LinearLayer>(this, "linear_layer_1", ThisDeviceNum ,NumEmbeddings,EmbeddingDim));
    WeightNode = this->SubLayers["linear_layer_1"]->GetLayerNodeName("Weight");
    CG->GetNode(WeightNode)->GetContent()->FillRandomValNormal();
    CG->GetNode(WeightNode)->Property.Set("Freeze", Freeze);
}

void EmbeddingLayer::AddEmbeddingNode(std::vector<size_t> InputShape, std::vector<size_t> InputData)
{
    EmbeddingChangeList.push_back({InputShape, InputData});
}

void EmbeddingLayer::FromPretrained(Tensor* PretrainedTensor)
{
    bool AssertFlag = (PretrainedTensor->shape.size() == 2)&&(PretrainedTensor->shape[0]==NumEmbeddings)&&(PretrainedTensor->shape[1]==EmbeddingDim);
    Log::Assert(AssertFlag, "Shape of Pretrained Tensor is NOT VALID\n");
    CG->GetNode(WeightNode)->AssignContent(PretrainedTensor);
}

std::vector<std::string> EmbeddingLayer::Forward(std::vector<std::string>InputNodeArray)
{
    std::vector<std::string>ReturnNodeList;
    for(size_t a = 0; a<EmbeddingChangeList.size();a++)
    {
        Tensor* OnehotTensor = Tensor::CreateOnehotTensor(EmbeddingChangeList[a].first, EmbeddingChangeList[a].second, NumEmbeddings, DeviceNum);
        std::string OnehotNodeName = CG->GetNodeidByOps(OpsType::Base,{});
        this->RegisterInputNode(OnehotNodeName, OnehotTensor->shape);
        CG->GetNode(OnehotNodeName)->AssignContent(OnehotTensor);
        std::string ReturnNodeName = OEAutoDiff::MatMul(CG, OnehotNodeName,WeightNode);
        ReturnNodeList.push_back(ReturnNodeName);
    }
    EmbeddingChangeList.clear();
    return ReturnNodeList;
}