#include "BaseOps.h"

void BaseOps::CommonInit(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG)
{
    this->OpsTypeName = OpsTypeName;
    this->Params = Params;
    CG = ParentCG;
}