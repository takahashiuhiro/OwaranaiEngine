#pragma once
#include "DynamicOptimizer/BaseDynamicOptimizer.h"
#include "DynamicOptimizer/SGD.h"

class Optimizer
{
public:
	static SGD CreateSGD(std::vector<DynamicTensor>Parameters,float LR = 0.001, float Momentum = 0, float WeightDecay = 0, float Dampening = 0, bool Nesterov = 0, bool Maximize = false);
};