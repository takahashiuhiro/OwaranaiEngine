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
#include "Code/AutomaticDifferentiation/Optimizer/SGDOptimizer.h"
int main() 
{
    size_t dd = 1;
    LinearSoftmaxLayer *m = new LinearSoftmaxLayer(nullptr, "gachi",dd,{1,2,3}, 0);
    m->RegisterInputNode("x", {1,3,2});
    auto ForwardRes = m->Forward({"x"});
    m->CG->GetNode("x")->GetContent()->FillArray(4);
    m->CG->ForwardDfs(ForwardRes[0]);
    m->CG->BackwardMultiBuildGraph(1);
    m->CG->GetNode(m->CG->GetDNodeid(ForwardRes[0]))->AssignContent(new Tensor({1,3,3},dd));
    m->CG->GetNode(m->CG->GetDNodeid(ForwardRes[0]))->GetContent()->FillArray(1.);
    m->CG->ForwardDfs(m->CG->GetDNodeid("gachi.layer_1.LinearWeight"));

    BaseOptimizer* qweddd = new SGDOptimizer();
    qweddd->Init(m->CG);
    qweddd->SyncTensorByCG();
    qweddd->TensorMap["gachi.layer_1.LinearWeight"].first->PrintData();
    qweddd->TensorMap["gachi.layer_1.LinearWeight"].second->PrintData();
    qweddd->Update();
    qweddd->ResTensorMap["gachi.layer_1.LinearWeight"]->PrintData();
}
