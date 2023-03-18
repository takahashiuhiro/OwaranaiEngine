#include "MathHelpers.h"


void AddVectorToVector(float* VectorInput, float* VectorOutput, float Weight, int Length)
{
    for(int a=0;a<Length;a++)
    {
        VectorOutput[a] += Weight*VectorInput[a];
    }
}

void MatrixGaussianElimination(float* InputMatrix, int Row, int Column)
{
    for(int a=0;a<Row;a++)
    {
        //遍历第几个主元
        for(int b=a;b<Row;b++)
        {
            if(!InputMatrix[b*Column+a])continue;
            else
            {
                if(b==a)break;
                for(int c=0;c<Column;c++)
                {
                    float TmpSwap = InputMatrix[b*Column + c];
                    InputMatrix[b*Column + c] = InputMatrix[a*Column + c];
                    InputMatrix[a*Column + c] = TmpSwap;
                }
                break;
            }
        }
        for(int b=0;b<Column;b++)
        {
            if(a==b)continue;
            InputMatrix[a*Column + b] /= InputMatrix[a*Column + a];
        }
        InputMatrix[a*Column+a] = 1.;
        //后面该多线程减了
        std::vector<std::thread> ThreadList;
        for(int b=0;b<Row;b++)
        {
            if(a == b||!InputMatrix[b*Column+a])continue;
            ThreadList.push_back(std::move(std::thread(AddVectorToVector, InputMatrix+a*Column, InputMatrix+b*Column, -1*InputMatrix[b*Column+a], Column)));
        }
        for(int b=0;b<ThreadList.size();b++)
        {
            ThreadList[b].join();
        }
    }
}