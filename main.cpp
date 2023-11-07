/**
一个简单的y=kx拟合，用来展示如何使用本框架
*/
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
    ComputationalGraph*m = new ComputationalGraph();
    std::string x = "x";
    std::vector<size_t>sp = {4,4};
    m->RegisterVariableNode(x,sp);
    m->GetNode(x)->AssignContent(new Tensor(sp, 0, {0.2035,  1.2959,  1.8101, -0.4644,1.5027, -0.3270,  0.5905,  0.6538,-1.5745,  1.3330, -0.5596, -0.6548,0.1264, -0.5080,  1.6420,  0.1992}));
    auto q = OEAutoDiff::Var(m,x,{0,1},false);

    m->ForwardDfs(q);
    m->GetNode(q)->PrintData();

}
