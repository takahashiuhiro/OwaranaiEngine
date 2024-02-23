#include "HyperElement.h"

void HE::r(int& Input){};
void HE::r(std::string& Input){};
void HE::r(float& Input){};
void HE::w(int Input){};
void HE::w(std::string Input){};
void HE::w(float Input){};
void HE::w(double Input){};

HEB::HEB(){HEBInit();}
HEB::HEB(int Input)
{
    HEBInit();
    w(Input);
}
HEB::HEB(std::string Input)
{
    HEBInit();
    w(Input);
}
HEB::HEB(float Input)
{
    HEBInit();
    w(Input);
}
HEB::HEB(double Input)
{
    HEBInit();
    w(Input);
}
void HEB::HEBInit(){ClassType = HEType::SHEB;}
bool HEB::CheckType(size_t InputType){return InputType == ElementType;}
void HEB::r(int& Input)
{
    Log::Assert(CheckType(ElementType), std::string("HEB TYPE ERROR, V TYPE IS NOT INT"));
    Input = InterVint;
}
void HEB::r(std::string& Input)
{
    Log::Assert(CheckType(ElementType), std::string("HEB TYPE ERROR, V TYPE IS NOT STRING"));
    Input = InterVstring;
}
void HEB::r(float& Input)
{
    Log::Assert(CheckType(ElementType), std::string("HEB TYPE ERROR, V TYPE IS NOT STRING"));
    Input = InterVfloat;
}
void HEB::w(int Input)
{
    InterVint = Input;
    ElementType = HEBType::INT;
}
void HEB::w(std::string Input)
{
    InterVstring = Input;
    ElementType = HEBType::STRING;
}
void HEB::w(float Input)
{
    InterVfloat = Input;
    ElementType = HEBType::FLOAT;
}
void HEB::w(double Input)
{
    InterVfloat = Input;
    ElementType = HEBType::FLOAT;
}