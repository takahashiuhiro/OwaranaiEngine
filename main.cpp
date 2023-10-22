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
#include "Code/AutomaticDifferentiation/Loss/MSELoss.h"
int main() 
{
    ComputationalGraph* w = new ComputationalGraph();

    w->RegisterVariableNode("x", {2,2});
    w->GetNode("x")->AssignContent(new Tensor({2,2}, 0, {1,-1,3,-4}));
    w->RegisterVariableNode("x1", {2,2});
    w->RegisterOpsCompleted("x1", {"x"}, OpsType::ReLU, Dict());
    w->BackwardMultiBuildGraph(1);

    w->GetNode(w->GetDNodeid("x1"))->AssignContent(new Tensor({2,2}, 0, {999,99,8888,7777}));

    w->ForwardDfs(w->GetDNodeid("x"));
    w->GetNode(w->GetDNodeid("x"))->PrintData();
}
