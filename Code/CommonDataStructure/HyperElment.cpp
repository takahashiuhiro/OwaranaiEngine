#include "HyperElement.h"

he::he(){heInit();}
he::he(int Input)
{
    heInit();
    w(Input);
}
he::he(std::string Input)
{
    heInit();
    w(Input);
}
he::he(float Input)
{
    heInit();
    w(Input);
}
he::he(double Input)
{
    heInit();
    w(Input);
}
void he::heInit(){}
bool he::CheckType(size_t InputType){return InputType == ElementType;}
int he::i()
{
    Log::Assert(CheckType(heType::INT), std::string("he TYPE ERROR, V TYPE IS NOT INT"));
    return InterVint;
}
std::string he::s()
{
    Log::Assert(CheckType(heType::STRING), std::string("he TYPE ERROR, V TYPE IS NOT STRING"));
    return InterVstring;
}
float he::f()
{
    Log::Assert(CheckType(heType::FLOAT), std::string("he TYPE ERROR, V TYPE IS NOT FLOAT"));
    return InterVfloat;
}
void he::w(int Input)
{
    InterVint = Input;
    ElementType = heType::INT;
}
void he::w(std::string Input)
{
    InterVstring = Input;
    ElementType = heType::STRING;
}
void he::w(float Input)
{
    InterVfloat = Input;
    ElementType = heType::FLOAT;
}
void he::w(double Input)
{
    InterVfloat = Input;
    ElementType = heType::FLOAT;
}
he::~he(){}
he he::operator + (he Other)const
{
    if(ElementType==heType::INT)
    {
        if(Other.ElementType==heType::INT)return he(InterVint + Other.InterVint);
        if(Other.ElementType==heType::STRING)return he(NumberToString(InterVint))+he(Other.InterVstring);
        if(Other.ElementType==heType::FLOAT)return he(InterVint + Other.InterVfloat);
    }
    if(ElementType==heType::STRING)
    {
        if(Other.ElementType==heType::STRING)return he(InterVstring + Other.InterVstring);
        if(Other.ElementType==heType::INT)return he(InterVstring + NumberToString(Other.InterVint));
    }
    if(ElementType == heType::FLOAT)
    {
        if(Other.ElementType==heType::INT)return he(InterVfloat + Other.InterVint);
        if(Other.ElementType==heType::FLOAT)return he(InterVfloat + Other.InterVfloat);
    }
    Log::Assert(false,std::string("he + TYPE ERROR, type tuple is ")+NumberToString(ElementType)+std::string(" ")+NumberToString(Other.ElementType));
    return he();
}
he he::operator * (int Other)const
{
    if(ElementType == heType::INT)return he(InterVint * Other);
    if(ElementType == heType::STRING)
    {
        std::string Res = "";
        for(int a=0;a<Other;a++)Res += InterVstring;
        return he(Res);
    }
    if(ElementType == heType::FLOAT)return he(InterVfloat * Other);
    Log::Assert(false,std::string("he *int TYPE ERROR, type tuple is ")+NumberToString(ElementType)+std::string(" ")+NumberToString(heType::INT));
    return he();
}
he he::operator * (double Other)const
{
    float tmp = Other;
    if(ElementType == heType::INT)return he(InterVint)*tmp;
    if(ElementType == heType::STRING)return he(InterVstring)*tmp;
    if(ElementType == heType::FLOAT)return he(InterVfloat)*tmp;
    Log::Assert(false,std::string("he *double TYPE ERROR, type tuple is ")+NumberToString(ElementType)+std::string(" ")+std::string(" double"));
    return he();
}
he he::operator * (float Other)const
{
    if(ElementType == heType::INT)return he(InterVint * Other);
    if(ElementType == heType::STRING)
    {
        std::string Res = "";
        int IntOther = Other;
        float FloatOther = Other - IntOther;
        int StrIndex = FloatOther*InterVstring.size();
        for(int a=0;a<IntOther;a++)Res += InterVstring;
        for(int a=0;a<StrIndex;a++)Res += InterVstring[a];
        return he(Res);
    }
    if(ElementType == heType::FLOAT)return he(InterVfloat * Other);
    Log::Assert(false,std::string("he *float TYPE ERROR, type tuple is ")+NumberToString(ElementType)+std::string(" ")+NumberToString(heType::FLOAT));
    return he();
}
he he::operator * (he Other)const
{
    if(ElementType==heType::INT)
    {
        if(Other.ElementType==heType::INT)return he(InterVint * Other.InterVint);
        if(Other.ElementType == heType::STRING)return he(Other.InterVstring)*InterVint;
        if(Other.ElementType==heType::FLOAT)return he(InterVint * Other.InterVfloat);
    }
    if(ElementType==heType::STRING)
    {
        if(Other.ElementType==heType::INT)return he(InterVstring)*Other.InterVint;
        if(Other.ElementType==heType::FLOAT)return he(InterVstring)*Other.InterVfloat;
    }
    if(ElementType == heType::FLOAT)
    {
        if(Other.ElementType==heType::INT)return he(InterVfloat * Other.InterVint);
        if(Other.ElementType == heType::STRING)return he(Other.InterVstring)*InterVfloat;
        if(Other.ElementType==heType::FLOAT)return he(InterVfloat * Other.InterVfloat);
    }
    Log::Assert(false,std::string("he * TYPE ERROR, type tuple is ")+NumberToString(ElementType)+std::string(" ")+NumberToString(Other.ElementType));
    return he();
}
he he::operator - (he Other)const
{
    if(ElementType==heType::INT)
    {
        if(Other.ElementType==heType::INT)return he(InterVint - Other.InterVint);
        if(Other.ElementType==heType::FLOAT)return he(InterVint - Other.InterVfloat);
    }
    if(ElementType == heType::FLOAT)
    {
        if(Other.ElementType==heType::INT)return he(InterVfloat - Other.InterVint);
        if(Other.ElementType==heType::FLOAT)return he(InterVfloat - Other.InterVfloat);
    }
    Log::Assert(false,std::string("he - TYPE ERROR, type tuple is ")+NumberToString(ElementType)+std::string(" ")+NumberToString(Other.ElementType));
    return he();
}
he he::operator / (he Other)const
{
    if(ElementType==heType::INT)
    {
        if(Other.ElementType==heType::INT)return he(InterVint / Other.InterVint);
        if(Other.ElementType==heType::FLOAT)return he(InterVint / Other.InterVfloat);
    }
    if(ElementType == heType::FLOAT)
    {
        if(Other.ElementType==heType::INT)return he(InterVfloat / Other.InterVint);
        if(Other.ElementType==heType::FLOAT)return he(InterVfloat / Other.InterVfloat);
    }
    Log::Assert(false,std::string("he / TYPE ERROR, type tuple is ")+NumberToString(ElementType)+std::string(" ")+NumberToString(Other.ElementType));
    return he();
}
void he::Set(he* ThisOther,he Other)
{
    ThisOther->ElementType = Other.ElementType;
    ThisOther->InterVint = Other.InterVint;
    ThisOther->InterVstring = Other.InterVstring;
    ThisOther->InterVfloat = Other.InterVfloat;
    ThisOther->MemoryArray = Other.MemoryArray;
    ThisOther->MemoryArrayUsefulLength = Other.MemoryArrayUsefulLength;
}
he he::operator = (he Other)const
{
    he Res;
    Res.Set(&Res,Other);
    return Res;
}
he& he::operator = (he Other)
{
    Set(this, Other);
    return (*this);
}
bool he::operator == (he Other)const
{
    if(ElementType!=Other.ElementType)return false;
    if(ElementType==heType::INT)return InterVint == Other.InterVint;
    if(ElementType==heType::STRING)return InterVstring == Other.InterVstring;
    if(ElementType==heType::FLOAT)return (InterVfloat - Other.InterVfloat)*(InterVfloat - Other.InterVfloat) < 1e-9;
    Log::Assert(false,std::string("he == is not define, type tuple : ")+NumberToString(ElementType)+std::string(" ")+NumberToString(Other.ElementType));
    return false;
}

he& he::operator [] (int Other)
{
    if(ElementType==heType::LIST)
    {
        return MemoryArray[Other];
    }
    Log::Assert(false,std::string("he [] is not define, type tuple : ")+NumberToString(ElementType)+std::string(" ")+std::string(" int"));
    return MemoryArray[0];
}

he& he::operator [] (he Other)
{
    if(ElementType==heType::LIST)
    {
        if(Other.ElementType == heType::INT)
        {
            return MemoryArray[Other.i()];
        }
    }
    Log::Assert(false,std::string("he [] is not define, type tuple : ")+NumberToString(ElementType)+std::string(" ")+NumberToString(Other.ElementType));
    return MemoryArray[0];
}
bool he::operator < (he Other)const
{
    if(ElementType==heType::INT)
    {
        if(Other.ElementType == heType::INT)return InterVint < Other.InterVint;
    }
    Log::Assert(false,std::string("he < is not define, type tuple : ")+NumberToString(ElementType)+std::string(" ")+NumberToString(Other.ElementType));
    return false;
}
bool he::operator > (he Other)const
{
    return (!operator<(Other))&&(!operator==(Other));
}
bool he::operator <= (he Other)const
{
    return !operator>(Other);
}
bool he::operator >= (he Other)const
{
    return !operator<(Other);
}

he he::NewList(int InputLength)
{
    he Res;
    Res.ElementType = heType::LIST;
    Res.MemoryArray.resize(InputLength);
    Res.MemoryArrayUsefulLength = InputLength;
    return Res;
}

he he::size()
{
    if(ElementType == heType::LIST)
    {
        return MemoryArrayUsefulLength;
    }
    Log::Assert(false,std::string("he size is not define, type tuple : ")+NumberToString(ElementType));
    return he(0);
}