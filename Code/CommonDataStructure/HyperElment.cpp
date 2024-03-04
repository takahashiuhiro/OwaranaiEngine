#include "HyperElement.h"

int HE::ri(){return 0;}
std::string HE::rs(){return "";}
float HE::rf(){return 0;}
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
int HEB::ri()
{
    Log::Assert(CheckType(HEBType::INT), std::string("HEB TYPE ERROR, V TYPE IS NOT INT"));
    return InterVint;
}
std::string HEB::rs()
{
    Log::Assert(CheckType(HEBType::STRING), std::string("HEB TYPE ERROR, V TYPE IS NOT STRING"));
    return InterVstring;
}
float HEB::rf()
{
    Log::Assert(CheckType(HEBType::FLOAT), std::string("HEB TYPE ERROR, V TYPE IS NOT FLOAT"));
    return InterVfloat;
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
HEB HEB::operator + (const HEB& Other)const
{
    if((Other.ElementType==HEBType::STRING)||(ElementType==HEBType::STRING))Log::Assert((ElementType == Other.ElementType),"HEB + TYPE ERROR, SREING CAN NOT + INT OR FLOAT");
    if(ElementType == HEBType::STRING)return HEB(InterVstring + Other.InterVstring);
    if(ElementType == HEBType::INT&&Other.ElementType==HEBType::INT)return HEB(InterVint + Other.InterVint);
    if(ElementType == HEBType::INT&&Other.ElementType==HEBType::FLOAT)return HEB(InterVint + Other.InterVfloat);
    if(ElementType == HEBType::FLOAT&&Other.ElementType==HEBType::INT)return HEB(InterVfloat + Other.InterVint);
    if(ElementType == HEBType::FLOAT&&Other.ElementType==HEBType::FLOAT)return HEB(InterVfloat + Other.InterVfloat);
}