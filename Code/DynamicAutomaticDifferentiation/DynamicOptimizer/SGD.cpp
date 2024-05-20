#include "SGD.h"

void SGD::ZeroGrad()
{
	for (size_t a = 0; a < Parameters.size(); a++)Parameters[a].Ops->GradOps = nullptr;
}

void SGD::Step()
{
	std::vector<DynamicTensor>GradList;
	if (std::abs(WeightDecay) > eps)
	{
		for (size_t a = 0; a < Parameters.size(); a++)GradList.push_back(Parameters[a].Grad().Copy() + Parameters[a].Copy() * WeightDecay);
	}
	else
	{
		for (size_t a = 0; a < Parameters.size(); a++)GradList.push_back(Parameters[a].Grad().Copy());
	}
	if (std::abs(Momentum) > eps)
	{
		if (UpdateTimes)
		{
			for (size_t a = 0; a < Parameters.size(); a++)ParametersMomentum[a] = ParametersMomentum[a] * Momentum + GradList[a] * (1 - Dampening);
		}
		else
		{
			for (size_t a = 0; a < Parameters.size(); a++)ParametersMomentum.push_back(GradList[a].Copy());
		}
		if (Nesterov)
		{
			for (size_t a = 0; a < Parameters.size(); a++)GradList[a] = GradList[a] + ParametersMomentum[a] * Momentum;
		}
		else
		{
			for (size_t a = 0; a < Parameters.size(); a++)GradList[a] = ParametersMomentum[a].Copy();
		}
	}
	UpdateTimes += 1;
	if (Maximize)for (size_t a = 0; a < Parameters.size(); a++)Parameters[a].Ops->TensorPointer = (Parameters[a].Copy() + GradList[a] * LR).Ops->TensorPointer;
	else for (size_t a = 0; a < Parameters.size(); a++)Parameters[a].Ops->TensorPointer = (Parameters[a].Copy() - GradList[a] * LR).Ops->TensorPointer;
}