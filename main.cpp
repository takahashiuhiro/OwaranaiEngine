#include <memory>
#include "Code/CommonDataStructure/HyperElement.h"
#include "Code/DynamicAutomaticDifferentiation/DynamicTensor.h"
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
#include <iostream>

DynamicTensor qwe(DynamicTensor& a, DynamicTensor& b, DynamicTensor& c)
{
    auto g = DynamicTensor::Add(a, b, 1);
    std::cout << "cacaca:" << &g << std::endl;
    return DynamicTensor::Add(g, c, 1);
}

int main() 
{
    DynamicTensor r = DynamicTensor::CreateVector({ 1 });
    DynamicTensor e = DynamicTensor::CreateVector({ 10 });
    DynamicTensor w = DynamicTensor::CreateVector({ 100 });
    DynamicTensor q = qwe(r, e, w);
    q.PrintData();
    std::cout << std::endl;
    std::cout << &r << std::endl;
    std::cout << &e << std::endl;
    std::cout << &w << std::endl;
    std::cout << &q << std::endl;
    std::cout << std::endl;
    std::cout << q.Ops.InputOpsList.size() << std::endl;
    std::cout << std::endl;
    std::cout << q.Ops.InputOpsList[0].InputOpsList[1].leafNode << std::endl;
    std::cout << "fengefu" << std::endl;
}
