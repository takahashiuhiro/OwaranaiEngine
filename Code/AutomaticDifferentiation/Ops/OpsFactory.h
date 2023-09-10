#pragma once
#include <iostream>
#include <memory>
#include <string>
#include "../../CommonDataStructure/Log.h"
#include "OpsType.h"
#include "AllOpsTypeHeader.h"
#include "../../CommonDataStructure/Dict.h"


class OpsFactory
{
public:

    static std::shared_ptr<BaseOps> GetOps(size_t OpsTypeid, Dict Params, ComputationalGraph* CG)
    {
        Log::Assert(OpsTypeid != OpsType::Base, std::string("Do Not Set Base Ops, id:"));
        if(OpsTypeid == OpsType::Add)
        {
            return std::make_shared<AddOps>(OpsTypeid, Params, CG);
        }
        if(OpsTypeid == OpsType::EleMul)
        {
            return std::make_shared<EleMulOps>(OpsTypeid, Params, CG);
        }
        if(OpsTypeid == OpsType::MatMul)
        {
            return std::make_shared<MatMulOps>(OpsTypeid, Params, CG);
        }
        if(OpsTypeid == OpsType::BroadCastTo)
        {
            return std::make_shared<BroadCastToOps>(OpsTypeid, Params, CG);
        }
        if(OpsTypeid == OpsType::Sum)
        {
            return std::make_shared<SumOps>(OpsTypeid, Params, CG);
        }
        if(OpsTypeid == OpsType::Softmax)
        {
            return std::make_shared<SoftmaxOps>(OpsTypeid, Params, CG);
        }
        Log::Assert(0, std::string("No Ops Be Set"));
    }
};
