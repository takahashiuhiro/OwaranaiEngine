#pragma once
#include "BaseDynamicOptimizer.h"

class SGD : public BaseDynamicOptimizer
{
public:
	std::vector<DynamicTensor>OptSolution;

	float lr = 0.01;

};