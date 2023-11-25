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
    Tensor* g = new Tensor({4,4},1,{
0.2035,  1.2959,  1.8101, -0.4644,
 1.5027, -0.3270,  0.5905,  0.6538,
-1.5745,  1.3330, -0.5596, -0.6548,
 0.1264, -0.5080,  1.6420,  0.1992
    });

    Tensor* yi = new Tensor({4,4},1,{
0.2035,  1.2959,  1.8101, -0.4644,
 1.5027, -0.3270,  0.5905,  0.6538,
-1.5745,  1.3330, -0.5596, -0.6548,
1,1,1,1.
    });

    ComputationalGraph * m = new ComputationalGraph();
    m->RegisterVariableNode("x",{4,4});
    m->GetNode("x")->AssignContent(g);

    std::string ss = OEAutoDiff::Tanh(m, "x");

    m->BackwardMultiBuildGraph(1);

    //Tensor* yi = new Tensor({4,4},1);
    //yi->FillArray(1.);
    m->GetNode(m->GetDNodeid(ss))->AssignContent(yi);

    m->ForwardDfs(ss);
    m->ForwardDfs(m->GetDNodeid("x"));
    //std::cout<<ss<<std::endl;
    m->GetNode(m->GetDNodeid("x"))->PrintData();
    m->GetNode(ss)->PrintData();

}
