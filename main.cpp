#include <memory>
#include "Code/CommonMathMoudle/Tensor.h"
#include "Code/AutomaticDifferentiation/ComputationalGraph.h"
#include "Code/AutomaticDifferentiation/Ops/OpsType.h"
#include "Code/AutomaticDifferentiation/AutoDiffCommon.h"
#include <cmath>
#include <fstream>
int main() 
{
    Tensor* q = new Tensor({2,5,4}, 0);
    //Tensor* w = new Tensor({1,2,5}, 0);
    q->FillArray(12);
    std::string qq = "ww.ee";
    //w->FillArray(1);
    //w->SaveToFile(qq);

    Tensor* w = Tensor::CreateTensorByLoadPath(qq);
    w->Matmul(q)->PrintData();
    w->PrintData();
}
