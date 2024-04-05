#include <memory>
#include "Code/CommonDataStructure/HyperElement.h"
#include "Code/DynamicAutomaticDifferentiation/DynamicTensor.h"
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
int main() 
{
    Tensor* s = Tensor::PositionalEncoding(4, 5,1);
    s->PrintData();
}
