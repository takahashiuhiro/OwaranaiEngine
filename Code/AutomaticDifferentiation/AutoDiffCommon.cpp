#include <stack>
#include "AutoDiffCommon.h"
#include "../CommonDataStructure/CommonFuncHelpers.h"
#include "../CommonDataStructure/Log.h"

ComputationalNodeInfo::ComputationalNodeInfo(std::string ProtoData)
{
    DecodeByString(ProtoData);
}

void ComputationalNodeInfo::DecodeByString(std::string ProtoData)
{
    bool InBlock = false;
    size_t NowIndex = 1;
    std::stack<bool>GapStack;
    for(size_t a = 1; a < ProtoData.size()-1;a++)
    {   
        if(ProtoData[a] == ',')
        {
            if(!InBlock)
            {
                std::string ThisBlock = ProtoData.substr(NowIndex, a - NowIndex);
                NowIndex = a+1;
                DecodeSingleBlock(ThisBlock);
            }
        }
        else if(ProtoData[a] == '{')
        {
            GapStack.push(0);
            InBlock = true;
        }
        else if(ProtoData[a] == '}')
        {
            Log::Assert(GapStack.size() > 0, std::string("Wrong ProtoData: ")+ProtoData);
            GapStack.pop();
            if(!GapStack.size())
            {
                InBlock = false;
            }
        }
    }
}

void ComputationalNodeInfo::DecodeSingleBlock(std::string ProtoData)
{
    std::string ThisBlockPre = "";
    std::string ThisBlockNxt = "";
    bool SignFlag = 0;
    bool FirstFlag = 1;
    for(size_t a = 0; a<ProtoData.size();a++)
    {
        if(ProtoData[a] == ':'&&FirstFlag)
        {
            SignFlag = 1;
            FirstFlag = 0;
            continue;
        }
        else
        {
            if(!SignFlag)ThisBlockPre += ProtoData[a];
            else ThisBlockNxt += ProtoData[a];
        }
    }
    if(ThisBlockPre == std::string("Name"))
    {
        Name = ThisBlockNxt;
    }
    else if(ThisBlockPre == std::string("Ops"))
    {
        Ops = StringToNumber<size_t>(ThisBlockNxt);
    }
    else if(ThisBlockPre == std::string("ExtraInfo"))
    {
        ExtraInfo = ThisBlockNxt;
    }
    else if(ThisBlockPre == std::string("D"))
    {
        D = StringToNumber<size_t>(ThisBlockNxt);
    }
}

std::string ComputationalNodeInfo::DumpToString()
{
    std::string ReturnStr = "{";
    ReturnStr+=BlockDumpToString("Name");
    ReturnStr+=BlockDumpToString("D");
    ReturnStr+=BlockDumpToString("Ops");
    ReturnStr+=BlockDumpToString("ExtraInfo");
    return ReturnStr + std::string("}");
}

std::string ComputationalNodeInfo::BlockDumpToString(std::string BlockString)
{
    std::string ReturnStr = BlockString + std::string(":");
    if(BlockString == std::string("Name"))
    {
        ReturnStr += Name;
    }
    else if(BlockString == std::string("ExtraInfo"))
    {
        ReturnStr += ExtraInfo;
    }
    else if(BlockString == std::string("Ops"))
    {
        ReturnStr += NumberToString(Ops);
    }
    else if(BlockString == std::string("D"))
    {
        ReturnStr += NumberToString(D);
    }
    return ReturnStr + std::string(",");
}