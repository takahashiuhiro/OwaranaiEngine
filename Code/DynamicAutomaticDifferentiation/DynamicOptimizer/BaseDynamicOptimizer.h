#pragma once

#include "../DynamicTensor.h"

class BaseDynamicOptimizer
{
public:
	float eps = 1e-9;
	he Params;

	virtual void ZeroGrad() = 0;
	virtual void Step() = 0;
};