#include "LinearLayer.h"

LinearLayer::LinearLayer(BaseLayer* ParentThis,std::string ThisLayerName, size_t ThisDeviceNum, std::vector<size_t>WeightShape)
{
    this->CommonInit(ParentThis, ThisLayerName, ThisDeviceNum);
    this->RegisterWeightNode("LinearWeight", WeightShape);
    //test
    //CG->GetNode(this->GetLayerNodeName("LinearWeight"))->AssignContent(new Tensor(WeightShape,ThisDeviceNum));
    //CG->GetNode(this->GetLayerNodeName("LinearWeight"))->GetContent()->FillArray(4.);
}

std::vector<std::string> LinearLayer::Forward(std::vector<std::string>InputNodeArray)
{
    std::string LinearWeightNodeID = this->GetLayerNodeName("LinearWeight");
    std::string OutputNodeid = CG->GetNodeidByOps(OpsType::MatMul, {InputNodeArray[0], LinearWeightNodeID});
    CG->RegisterVariableNode(OutputNodeid);
    CG->RegisterOpsCompleted(OutputNodeid, {InputNodeArray[0], LinearWeightNodeID}, OpsType::MatMul, Dict());
    CG->GetCGOps(OutputNodeid)->AfterSettingShapeComputing();
    return {OutputNodeid};
}