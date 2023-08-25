#include <memory>
#include "Code/CommonMathMoudle/Tensor.h"
#include "Code/AutomaticDifferentiation/ComputationalGraph.h"
#include "Code/AutomaticDifferentiation/Ops/OpsType.h"
int main() 
{
    Tensor *t = new Tensor({2,2,3},0);
    t->FillArray(1.);
    t->SetV({0,1,1}, 1.5);
    t->SetV({1,0,2}, 8);
    t->SetV({1,1,0}, 18);
    //t->PrintData();
    t->Minimum(2)->PrintData();
}