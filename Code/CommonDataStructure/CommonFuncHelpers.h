#pragma once
#include <iostream>
#include <string>
#include <algorithm>

template<typename T>
std::string NumberToString(T Input)
{
    std::string ReturnString = "";
    while(Input)
    {
        ReturnString += '0' + Input%10;
        Input /= 10;
    }
    std::reverse(ReturnString.begin(), ReturnString.end());
    return ReturnString;
}

template<typename T>
T StringToNumber(std::string Input)
{
    T ReturnV = 0;
    for(size_t a = 0;a < Input.size();a++)
    {
        ReturnV = ReturnV*10 + Input[a] - '0';
    }
    return ReturnV;
}