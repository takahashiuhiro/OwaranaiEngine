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
    s[he(5)] = he("888");
    s[he("qw")] = he(15.77);
    s[he(654)] = he(15.77);
    std::cout<< s.DumpToString()<<std::endl;
    s.SplayDelete(he(654));
    std::cout<< s.DumpToString()<<std::endl;
    s[he(654)] = he(1999.77);
    std::cout<< s.DumpToString()<<std::endl;
    s[he(654)] = he("asdasd");
    std::cout<< s.DumpToString()<<std::endl;
    //s.SplayPrintForDebugArray();
    s[he("gg")] = s;
    std::cout<< s.DumpToString()<<std::endl;
    s[he(77)] = he::NewList();
    s[he(77)].append(he("66"));
    s[he(77)].append(he(66));
    s[he(77)].append(s);
    std::cout<< s.DumpToString()<<std::endl;
    //s[he("gg")].SplayPrintForDebugArray();
}
