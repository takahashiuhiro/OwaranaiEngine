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

    std::map<std::string, int> HyperparameterType;
    std::map<std::string, std::vector<int> >IntParams;
    std::map<std::string, std::vector<float> >FloatParams;
    std::map<std::string, std::vector<std::string> >StringParams;

    void Set(std::string ParamName, int ParamType, std::any ParamContent)
    {
        HyperparameterType[ParamName] = ParamType;
        if(ParamType == HyperparameterTypeConst::INT)
        {
            std::vector<int> CastParamContent = std::any_cast<std::vector<int> >(ParamContent);
            IntParams[ParamName] = CastParamContent;
        }
        else if(ParamType == HyperparameterTypeConst::FLOAT)
        {
            std::vector<float> CastParamContent = std::any_cast<std::vector<float> >(ParamContent);
            FloatParams[ParamName] = CastParamContent;
        }
        else if(ParamType == HyperparameterTypeConst::STRING)
        {
            std::vector<std::string> CastParamContent = std::any_cast<std::vector<std::string> >(ParamContent);
            StringParams[ParamName] = CastParamContent;
        }
    }

    template<typename T>
    T* Get(std::string ParamName)
    {
        /**把类型强转成对应指针返回出去*/
        if(HyperparameterType[ParamName] == HyperparameterTypeConst::INT)return reinterpret_cast<T*>(&IntParams[ParamName]);
        if(HyperparameterType[ParamName] == HyperparameterTypeConst::FLOAT)return reinterpret_cast<T*>(&FloatParams[ParamName]);
        if(HyperparameterType[ParamName] == HyperparameterTypeConst::STRING)return reinterpret_cast<T*>(&StringParams[ParamName]);
        return nullptr;
    }

};