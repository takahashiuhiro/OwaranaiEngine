#pragma once
#include "StdInclude.h"

struct HyperparameterTypeConst
{
    static const int INT = 0;
    static const int FLOAT = 1;
    static const int STRING = 2;
};

struct Hyperparameter
{
public:
    std::map<std::string, int> ParamtypeMapping;
    //int t = HyperparameterTypeConst::INT;
};