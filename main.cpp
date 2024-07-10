#include <memory>
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
#include <iostream>
#include "Code/OEDynamic.h"


int main()
{
	auto BiasTensor = DynamicTensor({3,3},0,0);
    BiasTensor.Fill(1.);
    BiasTensor = BiasTensor.Tril();

	Tensor* r = new Tensor({3,3},0,{1,2,3,4,5,6,7,8,9.});
	DynamicTensor q = DynamicTensor(std::shared_ptr<Tensor>(r), 1);

	auto e = q.MaskedFill(BiasTensor, 99);

	e.Backward(DynamicTensor(std::shared_ptr<Tensor>(r->Copy())));

	print(e);
	print(q.Grad());
}
