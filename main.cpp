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
    size_t dnum = 0;

    Tensor* g = new Tensor({4,1,4},dnum,{
0.2035,  1.2959,  1.8101, -0.4644,
 1.5027, -0.3270,  0.5905,  0.6538,
-1.5745,  1.3330, -0.5596, -0.6548,
 0.1264, -0.5080,  1.6420,  0.1992
    });
    g->PrintData();

    Tensor* yi = new Tensor({8,2},1,{
0.2035,  1.2959,  1.8101, -0.4644,
 1.5027, -0.3270,  0.5905,  0.6538,
-1.5745,  1.3330, -0.5596, -0.6548,
1,1,1,1.
    });

    
    LinearLayer *m = new LinearLayer(nullptr, "LinearLayer",dnum,4,5);//声明网络
    m->RegisterInputNode("x", {4,1,4});//注册输入节点
    m->CG->GetNode("x")->AssignContent(g);
    std::string ForwardRes = m->Forward({"x"})[0];
    m->CG->ForwardDfs(ForwardRes);
    m->CG->GetNode("LinearLayer.Weight")->PrintData();
    m->CG->GetNode(ForwardRes)->PrintData();
}
