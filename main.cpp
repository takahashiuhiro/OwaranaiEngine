#include <memory>
#include "Code/CommonDataStructure/HyperElement.h"
#include "Code/DynamicAutomaticDifferentiation/DynamicTensor.h"
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
int main() 
{
    DynamicTensor::CreateDynamicTensor({2,2,1},0,{5,8,56,44}).PrintData();
    DynamicTensor::CreateDynamicTensor({2,2,1},0,{5,8,56,44}).T().PrintData();
}
