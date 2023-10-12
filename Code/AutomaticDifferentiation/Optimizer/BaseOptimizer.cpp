#include "BaseOptimizer.h"

std::vector<std::string> BaseOptimizer::GetWeightUpdateNodes()
{
    return CG->GetNodesByProperty({"Weight","RequireGrad"},{"Freeze"});
}