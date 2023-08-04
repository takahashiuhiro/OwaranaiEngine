#include "BaseOps.h"

void BaseOps::CommonInit(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG)
{
    this->OpsTypeName = OpsTypeName;
    this->Params = Params;
    CG = ParentCG;
}

void BaseOps::ParamsDefinition()
{
    /**每个输入样本的常数权重.*/
    this->Params.Set("AddWeight", std::make_shared<AddWeightType>());
}