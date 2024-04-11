#include <memory>
#include "Code/CommonDataStructure/HyperElement.h"
#include "Code/DynamicAutomaticDifferentiation/DynamicTensor.h"
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
int main() 
{
    DynamicTensor r = DynamicTensor::CreateVector({ 2,3 });
    DynamicTensor e = DynamicTensor::Add(r, r, true);
    e.PrintData();
}
