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
#include "Code/AutomaticDifferentiation/Layers/EmbeddingLayer.h"
int main() 
{
    size_t dm = 1;
    EmbeddingLayer* emb = new EmbeddingLayer(nullptr,"emb",dm,4,2,{false,0});
    Tensor* q = new Tensor({4,2},dm,{1,2,3,4,5,6,7,8.});
    emb->FromPretrained(q);
    emb->AddEmbeddingNode({2,2},{1,2,3,0});
    auto w = emb->Forward({})[0];
    emb->CG->ForwardDfs(w);
    emb->CG->GetNode(w)->PrintData();
}
