#pragma once
#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>
#include <vector>

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
    bool NPFlag = 0;
    if(Input[0] == '-')
    {
        NPFlag = 1;
        Input = Input.substr(1, Input.size()-1);
    }
    int PointIndex = -1;
    for(int a=0;a<Input.size();a++)
    {
        if(Input[a] == '.')
        {
            PointIndex = a;
            break;
        }
    }
    if((typeid(T)==typeid(double)||typeid(T)==typeid(float))&&PointIndex!=-1)
    {
        std::string IntPart = Input.substr(0,PointIndex);
        std::string FloatPart = Input.substr(PointIndex + 1, Input.size()-1-PointIndex);
        float IntPartFloat = StringToNumber<int>(IntPart);
        float FloatPartFloat = StringToNumber<int>(FloatPart);
        for(int a=0;a<FloatPart.size();a++)FloatPartFloat/=10.;
        return (IntPartFloat + FloatPartFloat)*(1-2*NPFlag);
    }
    T ReturnV = 0;
    for(size_t a = 0;a < Input.size();a++)
    {
        ReturnV = ReturnV*10 + Input[a] - '0';
    }
    return ReturnV*(1-2*NPFlag);
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

void print(bool Input);
void print(int Input);
void print(std::string Input);
void print(float Input);
void print(double Input);
void print(const char* Input);
void print(size_t Input);
template<typename T>
void print(std::vector<T> Input)
{
    for(size_t a = 0;a<Input.size();a++)
    {
        print(Input[a]);
    }
}
template<typename T>
void print(T Input){Input.PrintData();}