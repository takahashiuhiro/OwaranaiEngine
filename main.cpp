#include <memory>
#include "Code/CommonDataStructure/HyperElement.h"
#include "Code/DynamicAutomaticDifferentiation/DynamicTensor.h"
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
int main() 
{
    DynamicTensor a = DynamicTensor::CreateVector({1.2,5});
    DynamicTensor b = DynamicTensor::CreateDynamicTensor({2},0,{3.2,5});
    a+DynamicTensor::CreateDynamicTensor({2},0,{3.2,5});
    auto sd = (DynamicTensor::CreateDynamicTensor({2,1},0,{888.2,5})+DynamicTensor::CreateDynamicTensor({3,1,2},0,{3.2,5,77,88,99,1010}));
    sd.PrintData();
}
