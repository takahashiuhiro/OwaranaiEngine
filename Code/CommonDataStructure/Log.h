#pragma once
#include <iostream>
#include <cassert>
#include <string>

class Log
{
public:
    static void Assert(bool JudgeFlag, std::string AssertInfo)
    {
        if(JudgeFlag)return;
        std::cout<<AssertInfo<<std::endl;
        assert(JudgeFlag);
    }
};