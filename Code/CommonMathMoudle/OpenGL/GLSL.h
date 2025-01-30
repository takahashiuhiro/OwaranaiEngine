#pragma once
#include <vector>
#include <string>

class GLSL
{
private:
    GLSL()
    {
        AddGLSLFun();
        AddGLSLComFun();
    }

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


    /**
     * 拼装函数
     */
    int FillRandomValBernoulliInCPP;
    /**
     * 原型函数
     */
    int AddArrayInCPP;
    int FillArrayInCPP;
    int AddInCPP;
    int AddScalarInCPP;
    int EleMulInCPP;
    int MulScalarInCPP;
    int MatmulInCPP;
    int TInCPP;
    int SumTensorDimInCPP;
    int MaximumOrMinimumTensorDimInCPP;
    int TensorSpliceInCPP;
    int GetUnitTensorInCPP;
    int EleExpInCPP;
    int EleInverseInCPP;
    int BroadCastToInCPP;
    int FillRandomValNormalInCPP;
    int GenerateSignTensorInCPP;
    int PowInCPP;
    int FillRandomValUniformInCPP;
    int FillOnehotDataInCPP;
    int TrigonometricFunctionsInCPP;
    int ArithmeticSequenceInCPP;
    int GenerateTrilOnesInCPP;
    int TransposeInCPP;

void AddGLSLComFun()
{
    FillRandomValBernoulliInCPP = -1;
}

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

RegFun(TInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutput {
    float Output[];
};
layout(std430, binding = 1) buffer bufferInput {
    float Input[];
};
layout(std430, binding = 2) buffer bufferMatrixShape {
    int MatrixShape[];
};
layout(std430, binding = 3) buffer bufferShapeCount {
    int ShapeCount;
};
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if(Index < ShapeCount)
    {
        int MatrixShapeCount = MatrixShape[0]*MatrixShape[1];
        int InputMatIndex = int(Index)%MatrixShapeCount;
        int BaseCount = int(Index) - InputMatIndex;
        int InputMatIndexFirst = InputMatIndex/MatrixShape[1];
        int InputMatIndexSecond = InputMatIndex%MatrixShape[1];
        Output[BaseCount + InputMatIndexSecond*MatrixShape[0] + InputMatIndexFirst] = Input[Index];
    }
}
)");

RegFun(SumTensorDimInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutputData {
    float OutputData[];
};
layout(std430, binding = 1) buffer bufferInputData {
    float InputData[];
};
layout(std430, binding = 2) buffer bufferInputShape {
    int InputShape[];
};
layout(std430, binding = 3) buffer bufferInputShapeLen {
    int InputShapeLen;
};
layout(std430, binding = 4) buffer bufferInputDim {
    int InputDim;
};
layout(std430, binding = 5) buffer bufferOutputShapeCount {
    int OutputShapeCount;
};
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if(Index < OutputShapeCount)
    {
        int OutputIndex[8];
        int OutputSizeTMP = int(Index);
        for(int a=InputShapeLen-1;a>=0;a--)
        {
            if(a != InputDim) 
            {
                OutputIndex[a] = OutputSizeTMP%InputShape[a];
                OutputSizeTMP /= InputShape[a];
            }
            else
            {
                OutputIndex[a] = 0;
            }
        }
        OutputData[Index] = 0;
        for(int a =0;a<InputShape[InputDim];a++)
        {
            int InputDimIndex = 0;
            int InputSizeTMP = 1;
            for(int b = InputShapeLen - 1;b>=0;b--)
            {
                if(b!=InputDim)InputDimIndex += InputSizeTMP*OutputIndex[b];
                else InputDimIndex += InputSizeTMP*a;
                InputSizeTMP*=InputShape[b];
            }
            OutputData[Index] += InputData[InputDimIndex];
        }
    }
}
)");

RegFun(MaximumOrMinimumTensorDimInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutputData {
    float OutputData[];
};
layout(std430, binding = 1) buffer bufferInputData {
    float InputData[];
};
layout(std430, binding = 2) buffer bufferInputShape {
    int InputShape[];
};
layout(std430, binding = 3) buffer bufferInputShapeLen {
    int InputShapeLen;
};
layout(std430, binding = 4) buffer bufferInputDim {
    int InputDim;
};
layout(std430, binding = 5) buffer bufferOutputShapeCount {
    int OutputShapeCount;
};
layout(std430, binding = 6) buffer bufferIsMaximum {
    int IsMaximum;
};
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if(Index < OutputShapeCount)
    {
        if(bool(IsMaximum))
        {
            OutputData[Index] = -1e9+7;
        }
        else
        {
            OutputData[Index] = 1e9+7;
        }
        int OutputIndex[8];
        int OutputSizeTMP = int(Index);
        for(int a=InputShapeLen-1;a>=0;a--)
        {
            if(a != InputDim) 
            {
                OutputIndex[a] = OutputSizeTMP%InputShape[a];
                OutputSizeTMP /= InputShape[a];
            }
            else
            {
                OutputIndex[a] = 0;
            }
        }
        for(int a =0;a<InputShape[InputDim];a++)
        {
            int InputDimIndex = 0;
            int InputSizeTMP = 1;
            for(int b = InputShapeLen - 1;b>=0;b--)
            {
                if(b!=InputDim)InputDimIndex += InputSizeTMP*OutputIndex[b];
                else InputDimIndex += InputSizeTMP*a;
                InputSizeTMP*=InputShape[b];
            }
            if(bool(IsMaximum))
            {
                OutputData[Index] = max(OutputData[Index], InputData[InputDimIndex]);
            }
            else
            {
                OutputData[Index] = min(OutputData[Index], InputData[InputDimIndex]);
            }
        }
    }
}
)");

RegFun(TensorSpliceInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutputData {
    float OutputData[];
};
layout(std430, binding = 1) buffer bufferInputDataFirst {
    float InputDataFirst[];
};
layout(std430, binding = 2) buffer bufferInputDataSecond {
    float InputDataSecond[];
};
layout(std430, binding = 3) buffer bufferShapeCount {
    int InputShapeFirst[];
};
layout(std430, binding = 4) buffer bufferInputShapeSecond {
    int InputShapeSecond[];
};
layout(std430, binding = 5) buffer bufferInputShapeLen {
    int InputShapeLen;
};
layout(std430, binding = 6) buffer bufferInputDim {
    int InputDim;
};
layout(std430, binding = 7) buffer bufferOutputShapeCount {
    int OutputShapeCount;
};
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if(Index >= OutputShapeCount)return;
    int RightShapeCount = 1;
    //算出指定维度右边的单元大小
    for(int a=InputDim + 1;a<InputShapeLen;a++)
    {
        RightShapeCount*= InputShapeFirst[a];
    }
    //算出指定维度的大小
    int InputDimCount = InputShapeFirst[InputDim] + InputShapeSecond[InputDim];
    int LeftDimCount = int(Index)/RightShapeCount;
    int NowDimCount = LeftDimCount%InputDimCount;
    int StrictLeftDimCount = LeftDimCount/InputDimCount;
    if(NowDimCount < InputShapeFirst[InputDim])
    {
        OutputData[Index] = InputDataFirst[Index - StrictLeftDimCount*InputShapeSecond[InputDim]*RightShapeCount];
    }
    else
    {
        OutputData[Index] = InputDataSecond[Index - (StrictLeftDimCount+1)*InputShapeFirst[InputDim]*RightShapeCount];
    }
}
)");

RegFun(GetUnitTensorInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutputData {
    float OutputData[];
};
layout(std430, binding = 1) buffer bufferInputShape {
    int InputShape[];
};
layout(std430, binding = 2) buffer bufferOutputShapeCount {
    int OutputShapeCount;
};
layout(std430, binding = 3) buffer bufferInputShapeLen {
    int InputShapeLen;
};
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if(Index >= OutputShapeCount)return;
    int MatrixShapeCount = InputShape[InputShapeLen - 2]*InputShape[InputShapeLen - 1];
    int MatrixIndex = int(Index)%MatrixShapeCount;
    if(MatrixIndex%InputShape[InputShapeLen - 2] == MatrixIndex/InputShape[InputShapeLen - 2])
    {
      OutputData[Index] = 1;
    }
}
)");

RegFun(EleExpInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutputData {
    float OutputData[];
};
layout(std430, binding = 1) buffer bufferOutputShape {
    int OutputShape;
};
layout(std430, binding = 2) buffer bufferBaseNum {
    float BaseNum;
};
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if(Index >= OutputShape)return;
    OutputData[Index] = pow(BaseNum, OutputData[Index]);
}
)");

RegFun(EleInverseInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutputData {
    float OutputData[];
};
layout(std430, binding = 1) buffer bufferOutputShape {
    int OutputShape;
};
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if(Index >= OutputShape)return;
    OutputData[Index] = 1./OutputData[Index];
}
)");

RegFun(BroadCastToInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutputData {
    float OutputData[];
};
layout(std430, binding = 1) buffer bufferInputData {
    float InputData[];
};
layout(std430, binding = 2) buffer bufferOutputShape {
    int OutputShape[];
};
layout(std430, binding = 3) buffer bufferInputShape {
    int InputShape[];
};
layout(std430, binding = 4) buffer bufferShapeLen {
    int ShapeLen;
};
layout(std430, binding = 5) buffer bufferOutputShapeCount {
    int OutputShapeCount;
};
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if(Index >= OutputShapeCount)return;
    int ShapeIndex[10];
    int NowIndex = int(Index);
    for(int a = ShapeLen - 1 ;a >= 0;a--)
    {
        ShapeIndex[a] = NowIndex%OutputShape[a];
        NowIndex = int(NowIndex/OutputShape[a]);
        if(OutputShape[a] > InputShape[a])ShapeIndex[a] = 0;
    }
    int FixedInputIndex = 0;
    for(int a = 0;a<ShapeLen;a++)
    {
        FixedInputIndex *= InputShape[a];
        FixedInputIndex += ShapeIndex[a];
    }
    OutputData[Index] = InputData[FixedInputIndex];
}
)");

RegFun(FillRandomValNormalInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutputData {
    float OutputData[];
};
layout(std430, binding = 1) buffer bufferOutputShapeCount {
    int  OutputShapeCount;
};
layout(std430, binding = 2) buffer bufferMeanV {
    float MeanV;
};
layout(std430, binding = 3) buffer bufferVarianceV {
    float VarianceV;  
};
layout(std430, binding = 4) buffer bufferSeed {
    uint Seed;
};
float rand(inout uint seed)
{
    seed ^= 2747636419u;
    seed *= 2654435769u;
    seed ^= (seed >> 16u);
    return float(seed & 0xFFFFFFFFu) / 4294967295.0;
}
// Box-Muller 生成正态随机数
float generateGaussian(uint baseSeed, float mean, float param)
{
    uint threadID = gl_GlobalInvocationID.x;
    uint seed = baseSeed ^ threadID;
    float u1 = rand(seed);
    float u2 = rand(seed);
    float z0 = sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
    return z0 * param + mean;
}
void main()
{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(OutputShapeCount))return;
    OutputData[idx] = generateGaussian(Seed, MeanV, VarianceV);
}
)");

RegFun(GenerateSignTensorInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutputData {
    float OutputData[];
};
layout(std430, binding = 1) buffer bufferOutputShapeCount {
    int OutputShapeCount;
};
layout(std430, binding = 2) buffer bufferSwitchValue {
    float SwitchValue;
};
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if(Index >= OutputShapeCount)return;
    if(OutputData[Index] > SwitchValue)OutputData[Index] = 1.;
    else OutputData[Index] = 0.;
}
)");

RegFun(PowInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutputData {
    float OutputData[];
};
layout(std430, binding = 1) buffer bufferOutputShapeCount {
    int OutputShapeCount;
};
layout(std430, binding = 2) buffer bufferExponent {
    float Exponent;
};
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if(Index >= OutputShapeCount)return;
    OutputData[Index] = pow(OutputData[Index], Exponent);
}
)");

RegFun(FillRandomValUniformInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutputData {
    float OutputData[];
};
layout(std430, binding = 1) buffer bufferOutputShapeCount {
    int  OutputShapeCount;
};
layout(std430, binding = 2) buffer bufferMeanV {
    float MinV;
};
layout(std430, binding = 3) buffer bufferVarianceV {
    float MaxV;  
};
layout(std430, binding = 4) buffer bufferSeed {
    uint Seed;
};
float rand(inout uint seed)
{
    seed ^= gl_GlobalInvocationID.x;
    seed ^= 2747636419u;
    seed *= 2654435769u;
    seed ^= (seed >> 16u);
    return float(seed & 0xFFFFFFFFu) / 4294967295.0;
}
void main()
{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(OutputShapeCount))return;
    float Zero2OneRes = rand(Seed);
    OutputData[idx] = MinV + (MaxV-MinV)*Zero2OneRes;
}
)");

RegFun(FillOnehotDataInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutputData {
    float OutputData[];
};
layout(std430, binding = 1) buffer bufferBaseShape {
    int BaseShape;
};
layout(std430, binding = 2) buffer bufferOnehotShape {
    int OnehotShape;
};
layout(std430, binding = 3) buffer bufferInputData {
    int InputData[];
};
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if(Index >= BaseShape)return;
    OutputData[int(Index)*OnehotShape+InputData[Index]] = 1;
}
)");

RegFun(TrigonometricFunctionsInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutputData {
    float OutputData[];
};
layout(std430, binding = 1) buffer bufferOutputShapeCount {
    int OutputShapeCount;
};
layout(std430, binding = 2) buffer bufferFunType {
    int FunType;
};
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if(Index >= OutputShapeCount)return;
    if(FunType == 0)OutputData[Index] = sin(OutputData[Index]);
    if(FunType == 1)OutputData[Index] = cos(OutputData[Index]);
}
)");

RegFun(ArithmeticSequenceInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutputData {
    float OutputData[];
};
layout(std430, binding = 1) buffer bufferOutputShape {
    int OutputShape[];
};
layout(std430, binding = 2) buffer bufferOutputShapeSize {
    int OutputShapeSize;
};
layout(std430, binding = 3) buffer bufferA1 {
    float A1;
};
layout(std430, binding = 4) buffer bufferArithmetic {
    float Arithmetic;
};
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    int OutputShapeCount = 1;
    for(int a=0;a<OutputShapeSize;a++)OutputShapeCount *= OutputShape[a];
    if(Index >= OutputShapeCount)return;
    int CurIndex = int(Index)%OutputShape[OutputShapeSize-1];
    OutputData[Index] = CurIndex*Arithmetic + A1;
}
)");

RegFun(GenerateTrilOnesInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutputData {
    float OutputData[];
};
layout(std430, binding = 1) buffer bufferOutputShapeCount {
    int OutputShapeCount;
};
layout(std430, binding = 2) buffer bufferRow {
    int Row;
};
layout(std430, binding = 3) buffer bufferCol {
    int Col;
};
layout(std430, binding = 4) buffer bufferDiagonal {
    int Diagonal;
};
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    if(Index >= OutputShapeCount)return;
    int TrueIndex = int(Index)%(Row*Col);
    int ThisRow = TrueIndex/Col;
    int ThisCol = TrueIndex%Col;
    if(ThisCol <= ThisRow + Diagonal)OutputData[Index] = 1;
    else OutputData[Index] = 0;
}
)");

RegFun(TransposeInCPP,R"(
#version 430
layout(local_size_x = 256, local_size_y = 1) in;
layout(std430, binding = 0) buffer bufferOutputData {
    float OutputData[];
};
layout(std430, binding = 1) buffer bufferInputData {
    float InputData[];
};
layout(std430, binding = 2) buffer bufferOutputShape {
    int OutputShape[];
};
layout(std430, binding = 3) buffer bufferOutputShapeSize {
    int OutputShapeSize;
};
layout(std430, binding = 4) buffer bufferFirstDim {
    int FirstDim;
};
layout(std430, binding = 5) buffer bufferSecondDim {
    int SecondDim;
};
void main() 
{
    uint Index = gl_GlobalInvocationID.x;
    int OutputShapeCount = 1;
    int LeftDim = min(FirstDim, SecondDim);//最左边的维度
    int RightDim = max(FirstDim, SecondDim);//最右边的维度
    int MidRightElement = 1;//所有的要变换的维度的元素个数
    int RightElement = 1;//比要变换的右边的维度还小的
    int MidElement = 1;//中间的维度
    for(int a=0;a<OutputShapeSize;a++)
    {
        OutputShapeCount*=OutputShape[a];
        if(a < LeftDim)continue;
        MidRightElement*=OutputShape[a];
        if(a > RightDim)RightElement*=OutputShape[a];
        if(a < RightDim&&a > LeftDim)MidElement*=OutputShape[a];
    }
    if(Index >= OutputShapeCount)return;
    //交换维度得到输入的shape
    int NowIndex = int(Index)%MidRightElement;
    int UseIndex = NowIndex/RightElement;//右边都是行向量
    int ReduIndex = NowIndex%RightElement;
    int ADim = UseIndex/(MidElement*OutputShape[RightDim]);
    int BDim = (UseIndex%(MidElement*OutputShape[RightDim]))/OutputShape[RightDim];
    int CDim = UseIndex%OutputShape[RightDim];
    int InputUseIndex = CDim*(MidElement*OutputShape[LeftDim]) + BDim*OutputShape[LeftDim] + ADim;
    int InputNowIndex = InputUseIndex*RightElement + ReduIndex;
    int InputIndex = InputNowIndex+(int(Index/MidRightElement))*MidRightElement;
    OutputData[Index] = InputData[InputIndex];
}
)");

}};
