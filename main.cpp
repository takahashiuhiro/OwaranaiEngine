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
	//Tensor* q = new Tensor({ 2,3 }, 0, { 1,2,3,4,5,6. });
	//Tensor* w = new Tensor({ 2,4 }, 0, { 1,2,3,4,5,6.,7,8 });
	//Tensor* e = new Tensor({ 2,5 }, 0, { 1,2,3,4,5,6.,7,8,9,10 });
	//
	//Tensor::TensorCat({ q,w,e }, 1)->PrintData();
	//print(Tensor::TensorCat({ q,w,e }, 0)->shape);

	DynamicTensor q(std::shared_ptr<Tensor>(new Tensor({ 2,3 }, 0, { 1,2,3,4,5,6. })), true);
	DynamicTensor w(std::shared_ptr<Tensor>(new Tensor({ 2,4 }, 0, { 1,2,3,4,5,6.,7,8 })), true);
	DynamicTensor e(std::shared_ptr<Tensor>(new Tensor({ 2,5 }, 0, { 1,2,3,4,5,6.,7,8,9,10 })), true);

	auto r = DynamicTensor::Cat({ q,w,e },1);
	print(r);
	r = r.Sum();

	r.Backward();
	print(r);
	print(q.Grad());
	print(w.Grad());
	print(e.Grad());
}
