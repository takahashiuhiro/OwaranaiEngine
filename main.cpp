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
	Tensor* q = new Tensor({ 2,3 }, 0, { 1,2,3,4,5,6. });
	Tensor* w = new Tensor({ 2,4 }, 0, { 1,2,3,4,5,6.,7,8 });
	Tensor* e = new Tensor({ 2,5 }, 0, { 1,2,3,4,5,6.,7,8,9,10 });

	Tensor::TensorCat({ q,w,e }, 1)->PrintData();
	//print(Tensor::TensorCat({ q,w,e }, 0)->shape);
}
