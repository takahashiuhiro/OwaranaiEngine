#include <memory>
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
#include <iostream>
#include "Code/OEDynamic.h"


int main()
{
	Tensor* q = (new Tensor({3,3}, 0, {1,2,3,4,5,6,7,8,9.}))->Tril(-1);
	q->PrintData();
}
