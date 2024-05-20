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
	DynamicTensor q = DynamicTensor(std::shared_ptr<Tensor>(new Tensor({ 2,3,2 }, 0, { 1,2,3,4,5,6,77,8,9,10,11,12. })),1);
	//q->FillArray(1);


	//auto r = q.Eleexp(M_E);

	auto r = q.Tanh();
	r = r.Sum();
	print(r);
		
	r.Backward();
	print(q.Grad());
}
