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
    Tensor* q = new Tensor({4,2},dm,{1,2,3,4,5,6,7,8.});
    EmbeddingLayer* emb = new EmbeddingLayer(nullptr,"emb",dm,4,2,{true,2});
    emb->AddEmbeddingNode({2},{1,2});
    auto w = emb->Forward({})[0];
    emb->CG->BackwardMultiBuildGraph(1);
    Tensor* ee = new Tensor({2,2}, dm);
    ee->FillArray(1.);
    emb->CG->GetNode(emb->CG->GetDNodeid(w))->AssignContent(ee);
    emb->CG->ForwardDfs(w);
    emb->CG->ForwardDfs(emb->CG->GetDNodeid(emb->WeightNode));

    std::cout<<"-----------debug"<<std::endl;
    emb->CG->GetNode(emb->WeightNode)->PrintData();
    emb->CG->ForwardDfs(emb->CG->GetDNodeid(emb->WeightNode));
    emb->CG->GetNode(w)->PrintData();
    emb->CG->GetNode(emb->CG->GetDNodeid(emb->WeightNode))->PrintData();
}
