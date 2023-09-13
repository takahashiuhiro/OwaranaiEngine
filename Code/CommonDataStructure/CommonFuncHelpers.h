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
