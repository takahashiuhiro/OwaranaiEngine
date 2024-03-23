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

    s[1.667] = 6;
    s[8.5] = 6;
    s["1.667"] = 6;
    s[1.667] = 8;
    s[9.55] = 7.8;
    s[1] = "qweqweqwe";
    s["ggg"] = "bigo";

    std::cout<< s.DumpToString()<<std::endl;
    he scopy = he::LoadFromString(s.DumpToString());
    std::cout<< scopy.DumpToString()<<std::endl;
    std::cout<< (scopy.DumpToString() == s.DumpToString())<<std::endl;

    he gg = he::NewList();
    gg.append(6);
    gg.append(6.888);
    gg.append("asdasdasd");
    std::cout<<gg.DumpToString()<<std::endl;
}
