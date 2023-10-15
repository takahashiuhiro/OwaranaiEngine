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

    BaseOptimizer* qweddd = new SGDOptimizer();
    qweddd->Init(m->CG);

    m->CG->GetNode("gachi.layer_1.LinearWeight")->PrintData();
    for(int a=1;a<3;a++)
    {
        m->CG->GetNode("x")->AssignContent(new Tensor({1,3,3},dd));
        m->CG->GetNode("x")->GetContent()->FillArray(4);
        m->CG->GetNode(m->CG->GetDNodeid(ForwardRes[0]))->AssignContent(new Tensor({1,3,3},dd));
        m->CG->GetNode(m->CG->GetDNodeid(ForwardRes[0]))->GetContent()->FillArray(a*1.);
        m->CG->ForwardDfs(m->CG->GetDNodeid("gachi.layer_1.LinearWeight"));
        qweddd->SyncTensorByCG();
        qweddd->Update();
        qweddd->SyncTensorToCG();
        m->CG->ClearWeightConstExclude();
        m->CG->GetNode("gachi.layer_1.LinearWeight")->PrintData();
    }
}
