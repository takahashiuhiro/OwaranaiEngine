#include <memory>
#include "Code/CommonMathMoudle/Tensor.h"
#include "Code/AutomaticDifferentiation/ComputationalGraph.h"
#include "Code/AutomaticDifferentiation/Ops/OpsType.h"
#include <cmath>
int main() 
{
    Tensor *t = new Tensor({1,2,1},0);
    t->FillArray(4.);
    t->SetV({0,1,0}, 1);
    t->PrintData();
    t->BroadCastTo({4,2,3})->PrintData();
}