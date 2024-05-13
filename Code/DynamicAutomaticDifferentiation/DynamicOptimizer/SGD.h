#pragma once
#include "BaseDynamicOptimizer.h"

class SGD : public BaseDynamicOptimizer
{
public:
	
	std::vector<DynamicTensor>Parameters;
	std::vector<DynamicTensor>ParametersMomentum;

	float LR = 0.001;
	float Momentum = 0;
	float WeightDecay = 0;
	float Dampening = 0;
	bool Nesterov = 0;
	bool Maximize = false;

	int UpdateTimes = 0;

	virtual void ZeroGrad();
	virtual void Step();
};