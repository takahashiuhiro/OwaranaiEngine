#include "LinearLayer.h"

LinearLayer::LinearLayer(BaseLayer* ParentThis,std::string ThisLayerName, size_t ThisDeviceNum, size_t InFeatures,size_t OutFeatures, bool Bias)
{
    this->CommonInit(ParentThis, ThisLayerName, ThisDeviceNum);
    this->InFeatures = InFeatures;
    this->OutFeatures = OutFeatures;
    this->Bias = Bias;
    this->RegisterWeightNode("Weight", {InFeatures,OutFeatures});
    Tensor* WeightTensor = new Tensor({InFeatures,OutFeatures},DeviceNum);
    WeightTensor->FillRandomValUniform(-std::sqrt(1./InFeatures), std::sqrt(1./InFeatures));
    this->CG->GetNode(this->GetLayerNodeName("Weight"))->AssignContent(WeightTensor);
    if(Bias)
    {
        this->RegisterWeightNode("Bias", {1,OutFeatures});
        Tensor* BiasTensor = new Tensor({1,OutFeatures},DeviceNum);
        BiasTensor->FillRandomValUniform(-std::sqrt(1/InFeatures), std::sqrt(1/InFeatures));
        this->CG->GetNode(this->GetLayerNodeName("Bias"))->AssignContent(BiasTensor);
    }
}

std::vector<std::string> LinearLayer::Forward(std::vector<std::string>InputNodeArray)
{
    std::string WeightNode = this->GetLayerNodeName("Weight");
    std::string ViewNode = OEAutoDiff::View(this->CG,InputNodeArray[0],{0, InFeatures},0);
    std::string MatMulNode = OEAutoDiff::MatMul(this->CG, ViewNode, WeightNode);
    if(!Bias)return {MatMulNode};
    std::string BiasNode = this->GetLayerNodeName("Bias");
    size_t BiasBroadCastDim = this->CG->GetNode(MatMulNode)->NodeContentShape[0];
    std::string BiasBroadCastNode = OEAutoDiff::BroadCastTo(this->CG, BiasNode, {BiasBroadCastDim, OutFeatures});
    std::string AddNode = OEAutoDiff::Add(this->CG, {{MatMulNode,1.},{BiasBroadCastNode,1}});
    auto OutputShape = this->CG->GetNode(InputNodeArray[0])->NodeContentShape;
    OutputShape[OutputShape.size()-1] = OutFeatures;
    std::string OutputNode = OEAutoDiff::View(this->CG, AddNode, OutputShape);
    return {OutputNode};
}
