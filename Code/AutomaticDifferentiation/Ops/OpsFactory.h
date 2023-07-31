
#pragma once
#include <iostream>
#include <memory>
#include <string>
#include "OpsType.h"
#include "../../CommonDataStructure/Dict.h"


class OpsFactory
{
public:
    template<typename ComputationalGraph>
    static std::shared_ptr<BaseOps<ComputationalGraph>> GetOps(size_t OpsTypeid, Dict Params, ComputationalGraph* CG)
    {
        assert(OpsTypeid != OpsType::Base && "Do Not Set Base Ops");
        if(OpsTypeid == OpsType::Add)
        {
            return std::make_shared<AddOps<ComputationalGraph>>(OpsTypeid, Params, CG);
        }

        assert(0 && "No Ops Be Set");
    }
};