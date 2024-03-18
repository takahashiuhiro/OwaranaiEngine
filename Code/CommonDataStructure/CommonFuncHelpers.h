#pragma once
#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>

/**把数字变成字符串.*/
template<typename T>
std::string NumberToString(T Input)
{
    int NPFlag = 0;
    if(Input < 0)
    {
        NPFlag = 1;
        Input = -Input;
    }
    int IntInput = Input;
    float FloatInput = Input - IntInput;
    std::string ReturnString = "";
    if(NPFlag)ReturnString += '-';
    //整数部分
    std::string IntReturnString = "";
    while(IntInput)
    {
        IntReturnString += '0' + IntInput%10;
        IntInput /= 10;
    }
    if(IntReturnString == "")IntReturnString = std::string("0");
    std::reverse(IntReturnString.begin(), IntReturnString.end());
    ReturnString += IntReturnString;
    //浮点数部分
    std::string FloatReturnString = "";
    if(typeid(Input)==typeid(float)||typeid(Input)==typeid(double))
    {
        FloatReturnString += '.';
        int FloatPartInt = FloatInput*1000000;
        while(FloatPartInt%10==0&&FloatPartInt>0)FloatPartInt/=10;
        FloatReturnString += NumberToString(FloatPartInt);
    }
    ReturnString += FloatReturnString;
    return ReturnString;
}

/**把字符串转成数字.*/
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

/**把字符串存入文件.*/
template<typename T = int>
void SaveToFileString(std::ofstream& OpenedFile, std::string InputString)
{
    std::streamsize InputStringLength = InputString.length();
    OpenedFile.write(reinterpret_cast<const char*>(&InputStringLength), sizeof(InputStringLength));
    OpenedFile.write(reinterpret_cast<const char*>(InputString.c_str()), InputString.length());
}

/**从二进制文件读取字符串.*/
template<typename T = int>
std::string LoadFromFileString(std::ifstream& OpenedFile)
{
    std::streamsize OutputStringLength = 0;
    OpenedFile.read(reinterpret_cast<char*>(&OutputStringLength), sizeof(OutputStringLength));
    char* Buffer = new char[OutputStringLength];
    OpenedFile.read(Buffer, OutputStringLength);
    std::string ReturnString(Buffer, OutputStringLength);
    return ReturnString;
}