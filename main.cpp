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

    Tensor* g = Tensor::CreateOnehotTensor({2,2}, {8,2,3,4},9,1);
    g->PrintData();
}
