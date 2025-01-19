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
    int MatmulInCPP;


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
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if (Index < Size) Output[Index] = InputFirst[Index] + InputSecond[Index];
}
)");

RegFun(FillArrayInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferInput {
    float Input[];
};
layout(std430, binding = 1) buffer bufferScalar {
    float Scalar;
};
layout(std430, binding = 2) buffer bufferSize {
    int Size;
};
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if (Index < Size) Input[Index] = Scalar;
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
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if (Index < HighDimSize)Output[Index] = HighDimInput[Index] + LowDimInput[Index%LowDimSize];
}
)");

RegFun(AddScalarInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutput {
    float Output[];
};
layout(std430, binding = 1) buffer bufferInput {
    float Input[];
};
layout(std430, binding = 2) buffer bufferScalar {
    float Scalar;
};
layout(std430, binding = 3) buffer bufferSize {
    int Size;
};
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if (Index < Size) Output[Index] = Input[Index] + Scalar;
}
)");

RegFun(EleMulInCPP,R"(
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
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if (Index < HighDimSize)Output[Index] = HighDimInput[Index] * LowDimInput[Index%LowDimSize];
}
)");

RegFun(MulScalarInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutput {
    float Output[];
};
layout(std430, binding = 1) buffer bufferInput {
    float Input[];
};
layout(std430, binding = 2) buffer bufferScalar {
    float Scalar;
};
layout(std430, binding = 3) buffer bufferSize {
    int Size;
};
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if (Index < Size) Output[Index] = Input[Index] * Scalar;
}
)");

RegFun(MatmulInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutput {
    float Output[];
};
layout(std430, binding = 1) buffer bufferOutputBatchShape {
    int OutputBatchShape[];
};
layout(std430, binding = 2) buffer bufferOutputMatrixShape {
    int OutputMatrixShape[];
};
layout(std430, binding = 3) buffer bufferInputFirst {
    float InputFirst[];
};
layout(std430, binding = 4) buffer bufferInputFirstBatchShape {
    int InputFirstBatchShape[];
};
layout(std430, binding = 5) buffer bufferInputFirstMatrixShape {
    int InputFirstMatrixShape[];
};
layout(std430, binding = 6) buffer bufferInputSecond {
    float InputSecond[];
};
layout(std430, binding = 7) buffer bufferInputSecondBatchShape {
    int InputSecondBatchShape[];
};
layout(std430, binding = 8) buffer bufferInputSecondMatrixShape {
    int InputSecondMatrixShape[];
};
layout(std430, binding = 9) buffer bufferBatchShapeLen {
    int BatchShapeLen;
};
layout(std430, binding = 10) buffer bufferOutputShapeCount {
    int OutputShapeCount;
};
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if (Index < OutputShapeCount)
    {
        int OutputBatchIndex[8];
        int OutputMatrixShapeCount = OutputMatrixShape[0]*OutputMatrixShape[1];
        int OutSizeTMP = int(Index)/OutputMatrixShapeCount;
        bool MatZero = bool(OutSizeTMP);
        for(int a=BatchShapeLen-1;a>=0;a--)
        {
            if(!MatZero)OutputBatchIndex[a] = 0;
            else
            {
              OutputBatchIndex[a] = OutSizeTMP%OutputBatchShape[a];
              OutSizeTMP /= OutputBatchShape[a];
            }
        }
        int InputFirstBatchIndex[8];
        for(int a=BatchShapeLen-1;a>=0;a--)
        {
          if(OutputBatchIndex[a] < InputFirstBatchShape[a])InputFirstBatchIndex[a] = OutputBatchIndex[a];
          else InputFirstBatchIndex[a] = 0;
        }
        int InputFirstMatrixShapeCount = InputFirstMatrixShape[0]*InputFirstMatrixShape[1];
        int InputSecondBatchIndex[8];
        for(int a=BatchShapeLen-1;a>=0;a--)
        {
          if(OutputBatchIndex[a] < InputSecondBatchShape[a])InputSecondBatchIndex[a] = OutputBatchIndex[a];
          else InputSecondBatchIndex[a] = 0;
        }
        int InputSecondMatrixShapeCount = InputSecondMatrixShape[0]*InputSecondMatrixShape[1];
        int InputFirstBase = 0;
        int InFirstTMP = InputFirstMatrixShapeCount;
        for(int a=BatchShapeLen-1;a>=0;a--)
        {
          InputFirstBase += InFirstTMP*InputFirstBatchIndex[a];
          InFirstTMP*=InputFirstBatchShape[a];
        }
        int InputSecondBase = 0;
        int InSecondTMP = InputSecondMatrixShapeCount;
        for(int a=BatchShapeLen-1;a>=0;a--)
        {
          InputSecondBase += InSecondTMP*InputSecondBatchIndex[a];
          InSecondTMP*=InputSecondBatchShape[a];
        }
        int OutputMatrixIndex = int(Index)%OutputMatrixShapeCount;
        int MatIndex[2] = {OutputMatrixIndex/OutputMatrixShape[1], OutputMatrixIndex%OutputMatrixShape[1]};
        Output[Index] = 0;
        for(int a=0;a<InputFirstMatrixShape[1];a++)
        {
            Output[Index] += InputFirst[InputFirstBase + MatIndex[0]*InputFirstMatrixShape[1] + a]*InputSecond[InputSecondBase + a*InputSecondMatrixShape[1] + MatIndex[1]];
        }
    }
}
)");

}};
