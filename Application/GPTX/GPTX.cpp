#include "GPTX.h"

GPTX::GPTX()
{
    TokenIdxTable = he::NewDict();
}

void GPTX::GenTokenIdxTable(std::string InputName)
{
    auto LoadData = LoadTxtFromFile(InputName);
    std::map<std::string, int>Table;
    for(auto&it:LoadData)
    {
        for(int a=0;a+3<it.size();a+=3)
        {
            std::string TMPStr = it.substr(a,3);
            if(Table.find(TMPStr)==Table.end())Table[TMPStr] = 0;
            Table[TMPStr] += 1;
        }
    }
    std::vector<std::pair<std::string, int>>TableVec;
    for(auto&it:Table)TableVec.push_back(it);

    std::sort(TableVec.begin(), TableVec.end(), [](std::pair<std::string, int>& a,std::pair<std::string, int>& b) {return a.second > b.second;});

    for(int a=0;a<TableVec.size();a++)
    {
        TokenIdxTable[TableVec[a].first] = a+2;
        TokenIdxTable[a+2] = TableVec[a].first;
    }
}