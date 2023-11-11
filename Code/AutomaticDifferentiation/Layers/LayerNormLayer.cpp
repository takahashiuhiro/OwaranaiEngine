#include "LayerNormLayer.h"

LayerNormLayer::LayerNormLayer(BaseLayer* ParentThis,std::string ThisLayerName, size_t ThisDeviceNum,std::vector<size_t>WeightShape,size_t UseNum, bool ElementwiseAffine, float eps)
{
    this->CommonInit(ParentThis,ThisLayerName,ThisDeviceNum);
    this->ElementwiseAffine = ElementwiseAffine;
    this->WeightShape = WeightShape;
    this->UseNum = UseNum;
    this->eps = eps;
    this->RegisterConstNode("eps", WeightShape);
    CG->GetNode(this->GetLayerNodeName("eps"))->GetContent()->FillArray(eps);
    if(ElementwiseAffine)
    {
        this->RegisterWeightNode("Weight", WeightShape);
        CG->GetNode(this->GetLayerNodeName("Weight"))->GetContent()->FillArray(1.);
        this->RegisterWeightNode("Bias", WeightShape);
        CG->GetNode(this->GetLayerNodeName("Bias"))->GetContent()->FillArray(0);
    }
}

std::vector<std::string> LayerNormLayer::Forward(std::vector<std::string>InputNodeArray)
{
    std::vector<size_t>MeanDims;
    for(size_t a = UseNum+1;a<WeightShape.size();a++)MeanDims.push_back(a);
    std::string MeanNode = OEAutoDiff::Mean(this->CG, InputNodeArray[0], MeanDims);
    std::string BCNode = OEAutoDiff::BroadCastTo(this->CG, MeanNode, WeightShape);
    std::string MinusNode = OEAutoDiff::Add(this->CG, {{InputNodeArray[0], 1.}, {BCNode, -1.}});
    std::string VarNode = OEAutoDiff::Var(this->CG, InputNodeArray[0],MeanDims,false);
    std::string VarBCNode = OEAutoDiff::BroadCastTo(this->CG, VarNode, WeightShape);
    std::string AddNode = OEAutoDiff::Add(this->CG, {{VarBCNode, 1.}, {this->GetLayerNodeName("eps"), 1.}});
    std::string PowNode = OEAutoDiff::Pow(this->CG, AddNode, -0.5);
    std::string EleMulNode = OEAutoDiff::EleMul(this->CG, MinusNode, PowNode);
    if(!ElementwiseAffine)return {EleMulNode};
    std::string WeightEleMulNode = OEAutoDiff::EleMul(this->CG, this->GetLayerNodeName("Weight"), EleMulNode);
    std::string BiasAddNode = OEAutoDiff::Add(this->CG, {{WeightEleMulNode,1}, {this->GetLayerNodeName("Bias"),1}});
    return {BiasAddNode};
}