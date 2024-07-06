#include <memory>
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
#include <iostream>
#include "Code/OEDynamic.h"


int main()
{
	std::vector<float>cont;
	std::vector<size_t>test_shape = {2,3,4,5};
	size_t summ=1;
	for(auto&it:test_shape)summ*=it;
	for(int a=0;a<summ;a++)cont.push_back(a+1);
	Tensor* qw = new Tensor(test_shape, 0, cont);

	DynamicTensor q(std::shared_ptr<Tensor>(qw), 1);
	auto ee = DynamicTensor(std::shared_ptr<Tensor>(qw->Transpose(1,2)));

	auto qq = q.Transpose(1,2);

	qq.Backward(ee);

	print(qq);
	print(q.Grad());

}
