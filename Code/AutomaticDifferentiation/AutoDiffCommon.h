#pragma once
#include <iostream>
#include <string>

//解析节点名称信息
struct ComputationalNodeInfo
{
    ComputationalNodeInfo(){};
    ComputationalNodeInfo(std::string ProtoData);

    std::string Name = "";
    size_t D = 0;
    size_t Ops = 0;//base算子不能实例化
    std::string ExtraInfo = "";

    void DecodeByString(std::string ProtoData);
    void DecodeSingleBlock(std::string ProtoData);
    std::string DumpToString();
    std::string BlockDumpToString(std::string BlockString);
};