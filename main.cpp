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
    he s = he::NewDict();
    he d = he::NewDict();
    s[he("a")] = he(56.77000);
    s[he("b")] = he(58);
    s[he("c")] = he("asdasd");
    s[he(1)] = he("ttt");
    s[he(6.77)] = he(-999);
    d[he("yy")] = s;
    d[he(1)] = he::NewDict();
    d[he(1)][he("tt")] = he(987);
    std::cout<<(d[he(1)][he("tt")] + s[he("b")]).DumpToString()<<std::endl;
}
