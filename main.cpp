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
#include "Code/AutomaticDifferentiation/ForwardFunction.h"
int main() 
{
    ComputationalGraph*m = new ComputationalGraph();
    std::string x = "x";
    std::vector<size_t> sp = {2,2,3};
    m->RegisterVariableNode(x,sp);
    auto q = OEAutoDiff::Mean(m,{x},{1,2});

    m->BackwardMultiBuildGraph(1);

    m->GetNode(x)->AssignContent(new Tensor(sp, 0, {2.,6,4,7,8,9,4,2,11,6,4545,55}));

    m->GetNode(m->GetDNodeid(q[0]))->AssignContent(new Tensor({2,1,1}, 0, {777,666}));

    m->ForwardDfs(m->GetDNodeid(x));

    m->GetNode(m->GetDNodeid(x))->PrintData();
}
