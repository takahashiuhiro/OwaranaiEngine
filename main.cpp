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

    ComputationalGraph *m = new ComputationalGraph();

    std::string x = "x";
    std::vector<size_t> sp = {1,3};

    m->RegisterVariableNode(x, sp);
    m->RegisterVariableNode(x+x,sp);
    m->RegisterOpsCompleted(x+x, {x}, OpsType::Pow,Dict());
    m->GetCGOps(x+x)->SetEleExponent(7);
    m->GetCGOps(x+x)->AfterSettingShapeComputing();
    m->RegisterVariableNode(x+x+x,sp);
    m->RegisterOpsCompleted(x+x+x, {x+x}, OpsType::Pow,Dict());
    m->GetCGOps(x+x+x)->SetEleExponent(2);
    m->GetCGOps(x+x+x)->AfterSettingShapeComputing();
    m->BackwardMultiBuildGraph(1);

    m->GetNode(x)->AssignContent(new Tensor(sp, 0,{1,2,3}));
    m->GetNode(m->GetDNodeid(x+x+x))->AssignContent(new Tensor(sp, 0,{1,1,16}));

    m->ForwardDfs(m->GetDNodeid(x));
    m->ForwardDfs(x+x+x);
    m->GetNode(x+x+x)->PrintData();
    m->GetNode(m->GetDNodeid(x))->PrintData();
}
