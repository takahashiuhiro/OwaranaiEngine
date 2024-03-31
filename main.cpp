#include <memory>
#include "Code/CommonDataStructure/HyperElement.h"
#include "Code/DynamicAutomaticDifferentiation/DynamicTensor.h"
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
int main() 
{
    auto q = DynamicTensor::CreateDynamicTensor({1,2}, 0, {5,8.});
    auto w = DynamicTensor::CreateDynamicTensor({1,2}, 0, {5,9.});
    q->Params["requires_grad"] = 1;
    w->Params["requires_grad"] = 0;
    auto e = DynamicTensor::Add(q.get(),w.get(),1);
    e->InputList[0]->PrintData();
    w->OutputList[0]->PrintData();
}
