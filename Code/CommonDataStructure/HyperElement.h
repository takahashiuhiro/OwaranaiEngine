#include <iostream>
#include <string>
#include <functional>
#include "Log.h"
#include "CommonFuncHelpers.h"


struct heType
{
    static const size_t NONE = 0;
    static const size_t INT = 1;
    static const size_t STRING = 2;
    static const size_t FLOAT = 3;
};

struct he
{
    size_t ElementType = 0;
    int InterVint;
    std::string InterVstring;
    float InterVfloat;

    he();
    he(int Input);
    he(std::string Input);
    he(float Input);
    he(double Input);

    void heInit();
    bool CheckType(size_t InputType);
    int i();
    std::string s();
    float f();
    void w(int Input);
    void w(std::string Input);
    void w(float Input);
    void w(double Input);

    he operator + (he Other)const;
    he operator - (he Other)const;
    he operator * (he Other)const;
    he operator * (int Other)const;
    he operator * (double Other)const;
    he operator * (float Other)const;
    he operator / (he Other)const;
};