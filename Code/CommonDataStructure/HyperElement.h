#include <iostream>
#include <string>
#include <functional>
#include "Log.h"
#include "CommonFuncHelpers.h"

struct HEType
{
    static const size_t SHE = 0;
    static const size_t SHEB = 1;
};

struct HE
{
    size_t ClassType = HEType::SHE;
    /**↓是HEB函数.*/
    virtual void r(int& Input);
    virtual void r(std::string& Input);
    virtual void r(float& Input);
    virtual void w(int Input);
    virtual void w(std::string Input);
    virtual void w(float Input);
    virtual void w(double Input);
    /**↑是HEB函数.*/
};

struct HEBType
{
    static const size_t NONE = 0;
    static const size_t INT = 1;
    static const size_t STRING = 2;
    static const size_t FLOAT = 3;
};

struct HEB:public HE
{
    size_t ElementType = 0;
    int InterVint;
    std::string InterVstring;
    float InterVfloat;

    HEB();
    HEB(int Input);
    HEB(std::string Input);
    HEB(float Input);
    HEB(double Input);

    void HEBInit();
    bool CheckType(size_t InputType);
    virtual void r(int& Input);
    virtual void r(std::string& Input);
    virtual void r(float& Input);
    virtual void w(int Input);
    virtual void w(std::string Input);
    virtual void w(float Input);
    virtual void w(double Input);
};