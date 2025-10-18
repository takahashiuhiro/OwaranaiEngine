#pragma once
#include "DynamicOptimizer/BaseDynamicOptimizer.h"
#include "DynamicOptimizer/SGD.h"

namespace OwaranaiEngine
{

class Optimizer
{
public:
	static SGD CreateSGD(std::vector<DynamicTensor>Parameters,float LR = 0.001, float Momentum = 0, float WeightDecay = 0, float Dampening = 0, bool Nesterov = 0, bool Maximize = false);
};

SGD Optimizer::CreateSGD(std::vector<DynamicTensor>Parameters,float LR, float Momentum, float WeightDecay, float Dampening, bool Nesterov, bool Maximize)
{
	SGD ResOptimizer;
	ResOptimizer.Parameters = Parameters;
	ResOptimizer.LR = LR;
	ResOptimizer.Momentum = Momentum;
	ResOptimizer.WeightDecay = WeightDecay;
	ResOptimizer.Dampening = Dampening;
	ResOptimizer.Nesterov = Nesterov;
	ResOptimizer.Maximize = Maximize;
	return ResOptimizer;
}

}