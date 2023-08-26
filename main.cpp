#include <memory>
#include "Code/CommonMathMoudle/Tensor.h"
#include "Code/AutomaticDifferentiation/ComputationalGraph.h"
#include "Code/AutomaticDifferentiation/Ops/OpsType.h"
#include <cmath>
int main() 
{
    Tensor *t = new Tensor({2,3},1);
    t->FillArray(4.);
    t->SetV({1,1}, 1);
    t->EleExp(M_E)->PrintData();
}