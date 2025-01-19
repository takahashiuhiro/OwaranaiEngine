#pragma once
#include <vector>
#include <string>

class GLSL
{
private:
    GLSL(){AddGLSLFun();}

public:
    GLSL(const GLSL&) = delete;
    GLSL& operator=(const GLSL&) = delete;

    static GLSL& I() 
    {
        static GLSL instance; // 静态局部变量，线程安全
        return instance;
    }

    std::vector<std::string> GLSLFun;
    int GLSLFunNum = 0;
    char* GetFunStr(int FunNum){return GLSLFun[FunNum].data();}
    void RegFun(int& InputRegFun, std::string InputFunContent)
    {
        InputRegFun = GLSLFunNum++;
        GLSLFun.push_back(InputFunContent);
    }

    int AddArrayInCPP;
    int FillArrayInCPP;
    int AddInCPP;
    int AddScalarInCPP;
    int EleMulInCPP;
    int MulScalarInCPP;


void AddGLSLFun()
{

RegFun(AddArrayInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutput {
    float Output[];
};
layout(std430, binding = 1) buffer bufferInputFirst {
    float InputFirst[];
};
layout(std430, binding = 2) buffer bufferInputSecond {
    float InputSecond[];
};
layout(std430, binding = 3) buffer bufferSize {
    int Size;
};
void main() {
    uint Index = gl_GlobalInvocationID.x;
    if (Index < Size) Output[Index] = InputFirst[Index] + InputSecond[Index];
}
)");

RegFun(AddInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutput {
    float Output[];
};
layout(std430, binding = 1) buffer bufferHighDimInput {
    float HighDimInput[];
};
layout(std430, binding = 2) buffer bufferHighDimSize {
    int HighDimSize;
};
layout(std430, binding = 3) buffer bufferLowDimInput {
    float LowDimInput[];
};
layout(std430, binding = 4) buffer bufferLowDimSize {
    int LowDimSize;
};
void main() {
    uint Index = gl_GlobalInvocationID.x;
    if (Index < HighDimSize)Output[Index] = HighDimInput[Index] + LowDimInput[Index%LowDimSize];
}
)");







}};
