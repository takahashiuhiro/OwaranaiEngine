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
#include "Code/AutomaticDifferentiation/Layers/EmbeddingLayer.h"
#include "Code/CommonDataStructure/HyperElement.h"
#include <functional>
#include <stack>
int main() 
{
    //float w;
    //a.r(w);
    //std::cout<<w<<std::endl;
    //std::hash<std::string> tt;
    //std::cout<<tt("987987asdasd")<<std::endl;
    he a = he::NewList(3);
    a[0] = he(3);
    a[2] = he("abc");
    he b = he::NewList(3);
    b[0] = a;
    b[2] = a;
    b[2][0] = he(4);
    std::cout<<(b[2][0]<=he(4))<<std::endl;
    std::cout<<(b[2][0]*b[2][2]).s()<<std::endl;
}
