#include "Optimizer.h"

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