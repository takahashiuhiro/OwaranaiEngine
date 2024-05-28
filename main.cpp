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

	DynamicTensor q(std::shared_ptr<Tensor>(new Tensor({ 4,4 }, 0, { 0.2035,  1.2959,  1.8101, -0.4644,1.5027, -0.3270,  0.5905,  0.6538,-1.5745,  1.3330, -0.5596, -0.6548,0.1264, -0.5080,  1.6420,  0.1992 })), true);
	//DynamicTensor w(std::shared_ptr<Tensor>(new Tensor({ 2,4 }, 0, { 1,2,3,4,5,6.,7,8 })), true);
	//DynamicTensor e(std::shared_ptr<Tensor>(new Tensor({ 2,5 }, 0, { 1,2,3,4,5,6.,7,8,9,10 })), true);
	//
	auto r = q.Var({ 1 },1);
	print(r);
	r = r.Sum();
	print(r);
	r.Backward();
	//print(r);
	//print(q.Grad());
	//print(w.Grad());
	//print(e.Grad());
	//auto r = q.GELU().Sum();
	//r.Backward();
	//
	//print(q.Grad());
	print(q.Grad());


	he qweqwe = he::NewList(std::vector<std::string>{ "hh", "yy", "ww" });
	print(qweqwe);
	std::vector<std::string>rr;
	qweqwe.v(rr);
	print(rr);
}
