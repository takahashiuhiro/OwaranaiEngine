#pragma once
#include <iostream>
#include <memory>
#include <string>
#include "OpsType.h"
#include "AllOpsTypeHeader.h"
#include "../../CommonDataStructure/Dict.h"


class OpsFactory
{
public:

    static std::shared_ptr<BaseOps> GetOps(size_t OpsTypeid, Dict Params, ComputationalGraph* CG)
    {
        assert(OpsTypeid != OpsType::Base && "Do Not Set Base Ops");
        if(OpsTypeid == OpsType::Add)
        {
            return std::make_shared<AddOps>(OpsTypeid, Params, CG);
        }

        assert(0 && "No Ops Be Set");
    }
};