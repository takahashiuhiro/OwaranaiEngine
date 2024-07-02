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
	std::vector<size_t>test_shape = {2,1,3};
	size_t summ=1;
	for(auto&it:test_shape)summ*=it;
	for(int a=0;a<summ;a++)cont.push_back(a+1);
	Tensor* q = new Tensor(test_shape, 0, cont);
	q->PrintData();
	//print("---");
	q->Transpose(0,2)->PrintData();

	//q->Transpose(0,2);
}
