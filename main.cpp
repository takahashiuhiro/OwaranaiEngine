#include <memory>
#include "Code/CommonMathMoudle/Tensor.h"
#include "Code/AutomaticDifferentiation/ComputationalGraph.h"
#include "Code/AutomaticDifferentiation/Ops/OpsType.h"
#include "Code/AutomaticDifferentiation/AutoDiffCommon.h"
#include <cmath>
#include <fstream>
#include "Code/AutomaticDifferentiation/Layers/BaseLayer.h"
#include "Code/AutomaticDifferentiation/Layers/LinearSoftmaxLayer.h"
#include "Code/CommonDataStructure/CommonFuncHelpers.h"
#include "Code/AutomaticDifferentiation/Optimizer/BaseOptimizer.h"
int main() 
{
    size_t dd = 1;
    LinearSoftmaxLayer *m = new LinearSoftmaxLayer(nullptr, "gachi",dd,{2,4,3}, 0);
    m->RegisterInputNode("x", {2,3,4});
    auto ForwardRes = m->Forward({"x"});
    m->CG->GetNode("x")->GetContent()->FillArray(4);
    //m->SaveToFile("tttt");
    //m->SubLayers["layer_1"]->SaveToFile("tttt1111");
    //m->LoadFromFile("tttt");
    //m->SubLayers["layer_1"]->LoadFromFile("tttt1111");
    m->CG->ForwardDfs(ForwardRes[0]);
    //m->CG->GetNode(ForwardRes[0])->PrintData();
    m->CG->BackwardMultiBuildGraph(1);
    m->CG->GetNode(m->CG->GetDNodeid(ForwardRes[0]))->AssignContent(new Tensor({2,3,3},dd));
    m->CG->GetNode(m->CG->GetDNodeid(ForwardRes[0]))->GetContent()->FillArray(1.);
    m->CG->ForwardDfs(m->CG->GetDNodeid("gachi.layer_1.LinearWeight"));
    //m->CG->GetNode(m->CG->GetDNodeid("gachi.layer_1.LinearWeight"))->PrintData();

    BaseOptimizer qweddd;
}
