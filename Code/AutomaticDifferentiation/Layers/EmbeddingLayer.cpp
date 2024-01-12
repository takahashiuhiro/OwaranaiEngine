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
    if(PaddingIdx.first)
    {   
        std::string ConstNode = CG->GetNodeidByOps(OpsType::Base,{});
        this->RegisterConstNode(ConstNode, {NumEmbeddings,EmbeddingDim});
        Tensor*AllOneTensor = new Tensor({1,EmbeddingDim},ThisDeviceNum);
        AllOneTensor->FillArray(1.);
        std::vector<float>ConstData;
        for(size_t a = 0;a<NumEmbeddings;a++)ConstData.push_back(a != PaddingIdx.second);
        Tensor*EmbTensor = new Tensor({NumEmbeddings, 1}, ThisDeviceNum, ConstData);
        CG->GetNode(GetLayerNodeName(ConstNode))->AssignContent(EmbTensor->Matmul(AllOneTensor));
        PaddingWeightNode = OEAutoDiff::EleMul(CG, WeightNode, GetLayerNodeName(ConstNode));
        delete AllOneTensor;
        delete EmbTensor;
    }
    else
    {
        PaddingWeightNode = WeightNode;
    }
}

EmbeddingLayer::EmbeddingLayer(BaseLayer* ParentThis,std::string ThisLayerName, Tensor* PretrainedTensor, std::pair<bool, size_t> PaddingIdx,bool Freeze,std::pair<bool, float> MaxNorm, float NormType, bool ScaleGradByFreq, bool Sparse)
{
    bool AssertFlag = (PretrainedTensor->shape.size() == 2);
    Log::Assert(AssertFlag, "Shape of Pretrained Tensor !== 2\n");
    size_t NumEmbeddings = PretrainedTensor->shape[0];
    size_t EmbeddingDim = PretrainedTensor->shape[1];
    size_t ThisDeviceNum = PretrainedTensor->GetDeviceNum();
    this->CommonInit(ParentThis,ThisLayerName,ThisDeviceNum);
    this->PaddingIdx = PaddingIdx;
    this->Freeze = Freeze;
    this->MaxNorm = MaxNorm;
    this->NormType = NormType;
    this->ScaleGradByFreq = ScaleGradByFreq;
    this->Sparse = Sparse;
    this->RegisterLayer(std::make_shared<LinearLayer>(this, "linear_layer_1", ThisDeviceNum ,NumEmbeddings,EmbeddingDim));
    WeightNode = this->SubLayers["linear_layer_1"]->GetLayerNodeName("Weight");
    CG->GetNode(WeightNode)->AssignContent(PretrainedTensor);
    CG->GetNode(WeightNode)->Property.Set("Freeze", Freeze);
    if(PaddingIdx.first)
    {
        std::string ConstNode = CG->GetNodeidByOps(OpsType::Base,{});
        this->RegisterConstNode(ConstNode, {NumEmbeddings,EmbeddingDim});
        std::string LayerConstNode = GetLayerNodeName(ConstNode);
        Tensor*AllOneTensor = new Tensor({1,EmbeddingDim},ThisDeviceNum);
        AllOneTensor->FillArray(1.);
        std::vector<float>ConstData;
        for(size_t a = 0;a<NumEmbeddings;a++)ConstData.push_back(a != PaddingIdx.second);
        Tensor* EmbTensor = new Tensor({NumEmbeddings, 1}, ThisDeviceNum, ConstData);
        Tensor* EmbZeroTensor = EmbTensor->Matmul(AllOneTensor);
        delete AllOneTensor;
        delete EmbTensor;
        CG->GetNode(LayerConstNode)->AssignContent(EmbZeroTensor);
        Tensor* EmbAllTensor = EmbZeroTensor->Copy();
        EmbAllTensor->FillArray(1.);
        Tensor* NegEmbZeroTensor = EmbZeroTensor->MulScalar(-1.);
        Tensor* EmbOneTensor = EmbAllTensor->Add(NegEmbZeroTensor);
        delete NegEmbZeroTensor;
        delete EmbAllTensor;
        Tensor* EmbPretrained = EmbOneTensor->EleMul(PretrainedTensor);
        delete EmbOneTensor;
        std::string TMPNode = OEAutoDiff::EleMul(CG, WeightNode, GetLayerNodeName(ConstNode));
        std::string ConstNodeEMB = CG->GetNodeidByOps(OpsType::Base,{});
        this->RegisterConstNode(ConstNodeEMB, {NumEmbeddings,EmbeddingDim});
        std::string LayerConstNodeEMB = GetLayerNodeName(ConstNodeEMB);
        CG->GetNode(LayerConstNodeEMB)->AssignContent(EmbPretrained);
        PaddingWeightNode = OEAutoDiff::Add(CG, {{TMPNode,1.},{LayerConstNodeEMB,1}}); 
    }
    else
    {
        PaddingWeightNode = WeightNode;
    }
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
        std::string ReturnNodeName = OEAutoDiff::MatMul(CG, OnehotNodeName,PaddingWeightNode);
        ReturnNodeList.push_back(ReturnNodeName);
    }
    EmbeddingChangeList.clear();
    return ReturnNodeList;
}