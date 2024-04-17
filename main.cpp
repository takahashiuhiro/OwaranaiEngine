#include <memory>
#include "Code/CommonDataStructure/HyperElement.h"
#include "Code/DynamicAutomaticDifferentiation/DynamicTensor.h"
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
#include <iostream>
#include "Code/DynamicAutomaticDifferentiation/DynamicTensor.h"


int main() 
{
	DynamicTensor q = DynamicTensor(std::shared_ptr<Tensor>(new Tensor({ 1,2 }, 0, {2,3.})));
	DynamicTensor qw = DynamicTensor(std::shared_ptr<Tensor>(new Tensor({ 1,2 }, 0, { 4,7. })));
	auto rr = DynamicTensor::Add({ q,qw }, he(), 1);
	rr.Ops->TensorPointer->PrintData();
	std::cout << rr.Ops.get() << std::endl;
}
