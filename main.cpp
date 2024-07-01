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
	for(int a=0;a<210;a++)cont.push_back(a+1);
	Tensor* q = new Tensor({2,3,5,7}, 1, cont);
	q->PrintData();
	//print("---");
	q->Transpose(0,2)->PrintData();

	//q->Transpose(0,2);
}
