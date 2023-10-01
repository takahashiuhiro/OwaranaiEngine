#include <memory>
#include "Code/CommonMathMoudle/Tensor.h"
#include "Code/AutomaticDifferentiation/ComputationalGraph.h"
#include "Code/AutomaticDifferentiation/Ops/OpsType.h"
#include "Code/AutomaticDifferentiation/AutoDiffCommon.h"
#include <cmath>
#include <fstream>
#include "Code/AutomaticDifferentiation/Layers/BaseLayer.h"
#include "Code/AutomaticDifferentiation/Layers/LinearSoftmaxLayer.h"
int main() 
{
    LinearSoftmaxLayer *m = new LinearSoftmaxLayer(nullptr, "gachi",0,{2,4,3}, 0);

    m->CG->RegisterVariableNode("x", {2,3,4});
    auto ForwardRes = m->Forward({std::string("x")});

    m->CG->GetNode("x")->AssignContent(new Tensor({2,3,4},0));//对节点f的导数赋值张量
    m->CG->GetNode("x")->GetContent()->FillArray(4);

    //m->CG->PrintGraphAdjacencyList(1);
    m->CG->ForwardDfs(ForwardRes[0]);
    m->CG->GetNode(ForwardRes[0])->PrintData();
}
