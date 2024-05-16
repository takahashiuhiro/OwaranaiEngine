#include <memory>
#include "Code/CommonDataStructure/HyperElement.h"
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
#include <iostream>
#include "Code/DynamicAutomaticDifferentiation/DynamicTensor.h"
#include "Code/DynamicAutomaticDifferentiation/DynamicLayers/Linear.h"
#include "Code/DynamicAutomaticDifferentiation/Optimizer.h"

int main() 
{
	Tensor* q = new Tensor({ 2,3,4 },1);
	q->FillArray(1);
	auto g = q->GenerateSplitTensor({ 1,2 }, 1);
	for (auto we : g)
	{
		we->PrintData();
		print("--");
	}
}
