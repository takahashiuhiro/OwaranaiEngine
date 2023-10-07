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
int main() 
{
    size_t dd = 1;

    LinearSoftmaxLayer *m = new LinearSoftmaxLayer(nullptr, "gachi",dd,{2,4,3}, 0);

    m->CG->RegisterVariableNode("x", {2,3,4});
    auto ForwardRes = m->Forward({"x"});

    m->CG->GetNode("x")->AssignContent(new Tensor({2,3,4},dd));//对节点f的导数赋值张量
    m->CG->GetNode("x")->GetContent()->FillArray(4);

    //m->SaveToFile("tttt");
    //m->SubLayers["layer_1"]->SaveToFile("tttt1111");
    //m->LoadFromFile("tttt");
    m->SubLayers["layer_1"]->LoadFromFile("tttt1111");

    m->CG->ForwardDfs(ForwardRes[0]);
    m->CG->GetNode(ForwardRes[0])->PrintData();

    Tensor* fgfg = new Tensor({2,3},1);
    fgfg->FillRandomValNormal();
    fgfg->PrintData();

    Tensor* fgfg1 = new Tensor({2,3},0);
    fgfg1->FillRandomValNormal();
    fgfg1->PrintData();

}
