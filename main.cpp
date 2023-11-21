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
#include "Code/AutomaticDifferentiation/Layers/LayerNormLayer.h"
int main() 
{
    Tensor* g = new Tensor({3,5},1);
    g->FillRandomValNormal();
    g->PrintData();

    ComputationalGraph * m = new ComputationalGraph();
    m->RegisterVariableNode("x",{3,5});
    m->GetNode("x")->AssignContent(g);

    std::string gg = OEAutoDiff::Add(m,{{"x",1}});

    std::string ss = OEAutoDiff::Dropout(m,gg,0.5);

    m->SetEvalMode();

    m->ForwardDfs(ss);
    //std::cout<<ss<<std::endl;
    m->GetNode(ss)->PrintData();

}
