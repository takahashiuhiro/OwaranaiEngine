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
    Log::Assert(false,std::string("he + TYPE ERROR, type tuple is ")+heType::ToString(ElementType)+std::string(" ")+heType::ToString(Other.ElementType));
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
    Log::Assert(false,std::string("he *int TYPE ERROR, type tuple is ")+heType::ToString(ElementType)+std::string(" ")+NumberToString(heType::INT));
    return he();
}
he he::operator * (double Other)const
{
    float tmp = Other;
    if(ElementType == heType::INT)return he(InterVint)*tmp;
    if(ElementType == heType::STRING)return he(InterVstring)*tmp;
    if(ElementType == heType::FLOAT)return he(InterVfloat)*tmp;
    Log::Assert(false,std::string("he *double TYPE ERROR, type tuple is ")+heType::ToString(ElementType)+std::string(" ")+std::string(" double"));
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
    Log::Assert(false,std::string("he *float TYPE ERROR, type tuple is ")+heType::ToString(ElementType)+std::string(" ")+heType::ToString(heType::FLOAT));
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
    Log::Assert(false,std::string("he * TYPE ERROR, type tuple is ")+heType::ToString(ElementType)+std::string(" ")+heType::ToString(Other.ElementType));
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
    Log::Assert(false,std::string("he - TYPE ERROR, type tuple is ")+heType::ToString(ElementType)+std::string(" ")+heType::ToString(Other.ElementType));
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
    Log::Assert(false,std::string("he / TYPE ERROR, type tuple is ")+heType::ToString(ElementType)+std::string(" ")+heType::ToString(Other.ElementType));
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
    ThisOther->SplayRoot = Other.SplayRoot;
    ThisOther->HashInt = Other.HashInt;
    ThisOther->HashString = Other.HashString;
    ThisOther->HashFloat = Other.HashFloat;
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
    Log::Assert(false,std::string("he == is not define, type tuple : ")+heType::ToString(ElementType)+std::string(" ")+heType::ToString(Other.ElementType));
    return false;
}

he& he::operator [] (int Other)
{
    if(ElementType==heType::LIST)
    {
        return MemoryArray[Other];
    }
    Log::Assert(false,std::string("he [] is not define, type tuple : ")+heType::ToString(ElementType)+std::string(" ")+std::string(" int"));
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
    if(ElementType==heType::DICT)
    {
        if(Other.ElementType == heType::INT || Other.ElementType == heType::STRING ||Other.ElementType == heType::FLOAT)
        {
            return DictFromKtoV(Other);
        }
    }
    Log::Assert(false,std::string("he [] is not define, type tuple : ")+heType::ToString(ElementType)+std::string(" ")+heType::ToString(Other.ElementType));
    return MemoryArray[0];
}
bool he::operator < (he Other)const
{
    if(ElementType==heType::INT)
    {
        if(Other.ElementType == heType::INT)return InterVint < Other.InterVint;
    }
    Log::Assert(false,std::string("he < is not define, type tuple : ")+heType::ToString(ElementType)+std::string(" ")+heType::ToString(Other.ElementType));
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

std::string he::DumpToString()
{
    if(ElementType==heType::INT)
    {
        return NumberToString(InterVint);
    }
    if(ElementType==heType::STRING)
    {
        return std::string("\"")+InterVstring +std::string("\"");
    }
    if(ElementType == heType::FLOAT)
    {
        return NumberToString(InterVfloat);
    }
    if(ElementType == heType::LIST)
    {
        std::string ReturnString = "[";
        for(int a=0;a<MemoryArrayUsefulLength;a++)
        {
            ReturnString += MemoryArray[a].DumpToString();
            if(a<MemoryArrayUsefulLength-1)ReturnString += ",";
        }
        ReturnString += "]";
        return ReturnString;
    }
    if(ElementType == heType::DICT)
    {
        std::string ReturnString = "{";
        std::vector<int>SplayResult;
        SplayPrintForDebugDfs(SplayRoot, SplayResult);
        for(int a=0;a<SplayResult.size();a++)
        {
            ReturnString += MemoryArray[DictGetIndexKey(SplayResult[a])].DumpToString() + ":"+MemoryArray[DictGetIndexValue(SplayResult[a])].DumpToString();
            if(a<SplayResult.size()-1)ReturnString+=",";
        }
        ReturnString += "}";
        return ReturnString;
    }
    Log::Assert(false,std::string("he DumpToString is not define, type tuple : ")+heType::ToString(ElementType));
    return "";
}

he he::size()
{
    if(ElementType == heType::LIST)
    {
        return MemoryArrayUsefulLength;
    }
    Log::Assert(false,std::string("he size is not define, type tuple : ")+heType::ToString(ElementType));
    return he(0);
}

int he::hash(he Other)
{
    if(Other.ElementType == heType::INT)
    {
        return HashInt(Other.InterVint);
    }
    if(Other.ElementType == heType::STRING)
    {
        return HashString(Other.InterVstring);
    }
    if(Other.ElementType == heType::FLOAT)
    {
        return HashFloat(Other.InterVfloat);
    }
    Log::Assert(false,std::string("he hash is not define, type tuple : ")+heType::ToString(Other.ElementType));
    return 0;
}

he he::NewList(int InputLength)
{
    he Res;
    Res.ElementType = heType::LIST;
    Res.MemoryArray.resize(InputLength);
    Res.MemoryArrayUsefulLength = InputLength;
    return Res;
}

void he::append(he Other)
{
    if(ElementType == heType::LIST)
    {
        MemoryArray.push_back(Other);
        MemoryArrayUsefulLength +=1;
        return;
    }
    Log::Assert(false,std::string("he append is not define, type tuple : ")+heType::ToString(ElementType));
}

he he::NewDict()
{
    he Res;
    Res.ElementType = heType::DICT;
    return Res;
}

void he::DictApplyNewBlock(int BlockNum)
{
    for(int a=0;a<BlockNum;a++)
    {
        MemoryArray.emplace_back();
        MemoryArray.emplace_back();
        MemoryArray.emplace_back(-1);
        MemoryArray.emplace_back(-1);
        MemoryArray.emplace_back(-1);
    }
}

int he::DictGetNextIndex()
{
    MemoryArrayUsefulLength ++;
    if(DictMemoryIndexManager.empty())
    {
        DictApplyNewBlock(1);
        return MemoryArray.size()/5 - 1;
    }
    int res = DictMemoryIndexManager.top();
    DictMemoryIndexManager.pop();
    return res;
}

void he::DictSetMemoryBlockByIndex(int InputIndex,he InputKey, he InputValue, bool IsInit)
{
    MemoryArray[DictGetIndexKey(InputIndex)] = InputKey;
    MemoryArray[DictGetIndexValue(InputIndex)] = InputValue;
    if(!IsInit)return;
    MemoryArray[DictGetIndexLeft(InputIndex)] = he(-1);
    MemoryArray[DictGetIndexRight(InputIndex)] = he(-1);
    MemoryArray[DictGetIndexPre(InputIndex)] = he(-1);
}

int he::DictGetIndexKey(int InputIndex){return 5*InputIndex;}
int he::DictGetIndexValue(int InputIndex){return 5*InputIndex+1;}
int he::DictGetIndexLeft(int InputIndex){return 5*InputIndex+3;}
int he::DictGetIndexRight(int InputIndex){return 5*InputIndex+4;}
int he::DictGetIndexPre(int InputIndex){return 5*InputIndex+2;}

DictFindDfsInfo he::SplayInputFindDfs(int RootIndex, int InputKey)
{
    //std::cout<<"gogo::"<<RootIndex<<std::endl;
    int NowKey = hash(MemoryArray[DictGetIndexKey(RootIndex)]);
    //std::cout<<"gogo2::"<<MemoryArray[DictGetIndexKey(RootIndex)].DumpToString()<<std::endl;
    if(NowKey == InputKey)
    {
        return {1,RootIndex,-1};
    }
    if(InputKey < NowKey)
    {
        if(MemoryArray[DictGetIndexLeft(RootIndex)].i()==-1)return {0, RootIndex, 0};
        return SplayInputFindDfs(MemoryArray[DictGetIndexLeft(RootIndex)].i(),InputKey);
    }
    if(InputKey > NowKey)
    {
        if(MemoryArray[DictGetIndexRight(RootIndex)].i()==-1)return {0, RootIndex, 1};
        return SplayInputFindDfs(MemoryArray[DictGetIndexRight(RootIndex)].i(),InputKey);
    }
    return {0,0,0};
}

DictFindDfsInfo he::SplayFind(he InputKey)
{
    if(!MemoryArrayUsefulLength)return{0,-1,-1};
    int InputKeyHashV = hash(InputKey);
    DictFindDfsInfo ResIndex = SplayInputFindDfs(SplayRoot, InputKeyHashV);
    return ResIndex;
}

he& he::DictFromKtoV(he InputKey)
{
    DictFindDfsInfo Res = SplayFind(InputKey);
    if(Res.HasResult)
    {
        return MemoryArray[DictGetIndexValue(Res.MemoryIndex)];
    }
    else
    {
        int NewBlock = SplayInsert(InputKey, he());
        return MemoryArray[DictGetIndexValue(NewBlock)];
    }
}

int he::SplayInsert(he InputKey, he InputValue)
{
    DictFindDfsInfo FindRes = SplayFind(InputKey);
    if(FindRes.HasResult)
    {
        DictSetMemoryBlockByIndex(FindRes.MemoryIndex, InputKey, InputValue);
        return FindRes.MemoryIndex;
    }
    int NewBlock = DictGetNextIndex();
    DictSetMemoryBlockByIndex(NewBlock, InputKey, InputValue, 1);
    if(FindRes.MemoryIndex == -1)
    {
        SplayRoot = NewBlock;
        return NewBlock;
    }
    MemoryArray[DictGetIndexPre(NewBlock)] = he(FindRes.MemoryIndex);
    if(FindRes.IsLeft == 0)
    {
        MemoryArray[DictGetIndexLeft(FindRes.MemoryIndex)] = he(NewBlock);
    }
    if(FindRes.IsLeft == 1)
    {
        MemoryArray[DictGetIndexRight(FindRes.MemoryIndex)] = he(NewBlock);
    }
    Splay(NewBlock);
    return NewBlock;
}

void he::Splay(int InputIndex)
{
    //todo::旋转
}

void he::SplayPrintForDebugArray()
{
    std::cout<<"Root:: "<<SplayRoot<<std::endl;
    for(int a=0;a<MemoryArray.size()/5;a++)SplayPrintForDebugSingleNode(a);
}

void he::SplayPrintForDebugTree()
{
    std::cout<<"Root:: "<<SplayRoot<<std::endl;
    std::vector<int>Result;
    SplayPrintForDebugDfs(SplayRoot, Result);
    for(int a=0;a<Result.size();a++)SplayPrintForDebugSingleNode(a);
}
void he::SplayPrintForDebugDfs(int Root, std::vector<int>&Result)
{
    Result.push_back(Root);
    if(MemoryArray[DictGetIndexLeft(Root)].i() != -1)SplayPrintForDebugDfs(MemoryArray[DictGetIndexLeft(Root)].i(),Result);
    if(MemoryArray[DictGetIndexRight(Root)].i() != -1)SplayPrintForDebugDfs(MemoryArray[DictGetIndexRight(Root)].i(),Result);
}

void he::SplayPrintForDebugSingleNode(int Root)
{
    std::cout<<"Index:: "<<Root<<" ,Key:: "<<MemoryArray[DictGetIndexKey(Root)].DumpToString()<<" ,Hash:: "<<hash(MemoryArray[DictGetIndexKey(Root)])<<" ,Left:: ";
    if(MemoryArray[DictGetIndexLeft(Root)].i() == -1)std::cout<<"Null";
    else std::cout<<MemoryArray[DictGetIndexLeft(Root)].DumpToString();
    std::cout<<" ,Right:: ";
    if(MemoryArray[DictGetIndexRight(Root)].i() == -1)std::cout<<"Null";
    else std::cout<<MemoryArray[DictGetIndexRight(Root)].DumpToString();
    std::cout<<std::endl;
}

void he::SplayDelete(he InputKey)
{
    DictFindDfsInfo FindRes = SplayFind(InputKey);
    Log::Assert(FindRes.HasResult, std::string("Key is Not Existed::")+InputKey.DumpToString());
    int CurNode = FindRes.MemoryIndex;//待删节点
    int PreNodeIndex = MemoryArray[DictGetIndexPre(CurNode)].i();//待删节点的父节点
    int LeftNode = MemoryArray[DictGetIndexLeft(CurNode)].i();//待删节点的左子树
    int RightNode = MemoryArray[DictGetIndexRight(CurNode)].i();//待删节点的右子树
    bool IsDelRoot = PreNodeIndex == -1;//删除的节点是根节点吗
    if(IsDelRoot || MemoryArray[DictGetIndexLeft(PreNodeIndex)].i() == CurNode)//被删掉的节点是父节点的左子树
    {
        if(LeftNode == -1 && RightNode == -1)//被删的是叶子节点
        {
            if(IsDelRoot)SplayRoot = -1;//如果删除的是根节点，那树就删没了
            else MemoryArray[DictGetIndexLeft(PreNodeIndex)] = he(-1);//如果不是根节点，父节点的左子树得变为空
        }
        if(LeftNode != -1 && RightNode == -1)//待删节点的左子树非空，右子树空
        {
            if(IsDelRoot)//如果待删节点只有一边，那就直接删了换根
            {
                SplayRoot = LeftNode;
                MemoryArray[DictGetIndexPre(LeftNode)] = he(-1);
            }
            else
            {
                MemoryArray[DictGetIndexLeft(PreNodeIndex)] = he(LeftNode);
                MemoryArray[DictGetIndexPre(LeftNode)] = he(PreNodeIndex);
            }
        }
        if(LeftNode == -1 && RightNode != -1)//待删节点的左子树空，右子树非空
        {
            if(IsDelRoot)//如果待删节点只有一边，那就直接删了换根
            {
                SplayRoot = RightNode;
                MemoryArray[DictGetIndexPre(RightNode)] = he(-1);
            }
            else
            {
                MemoryArray[DictGetIndexLeft(PreNodeIndex)] = he(RightNode);
                MemoryArray[DictGetIndexPre(RightNode)] = he(PreNodeIndex);
            }
        }
        if(LeftNode != -1 && RightNode != -1)
        {
            //寻找右侧子树中的第一个左侧节点
            int RightRoot = RightNode;
            int FindRightLeft = -1;
            while(true)
            {
                if(RightRoot==-1)break;
                else
                {
                    FindRightLeft = RightRoot;
                    RightRoot = MemoryArray[DictGetIndexLeft(RightRoot)].i();
                }
            }
            //把删掉节点的左孩子的父节点设为右侧最左节点
            MemoryArray[DictGetIndexPre(LeftNode)] = he(FindRightLeft);
            MemoryArray[DictGetIndexLeft(FindRightLeft)] = he(LeftNode);
            //把删掉的右孩子代替删掉的节点
            if(IsDelRoot)
            {
                SplayRoot = RightNode;
                MemoryArray[DictGetIndexPre(RightNode)] = he(-1);
            }
            else
            {
                MemoryArray[DictGetIndexLeft(PreNodeIndex)] = he(RightNode);
                MemoryArray[DictGetIndexPre(RightNode)] = he(PreNodeIndex);
            }
        }
    }
    else
    {
        if(LeftNode == -1 && RightNode == -1)//被删的是叶子节点
        {
            MemoryArray[DictGetIndexRight(PreNodeIndex)] = he(-1);//如果不是根节点，父节点的右子树得变为空
        }
        if(LeftNode != -1 && RightNode == -1)//待删节点的左子树非空，右子树空
        {
            MemoryArray[DictGetIndexRight(PreNodeIndex)] = he(LeftNode);
            MemoryArray[DictGetIndexPre(LeftNode)] = he(PreNodeIndex);
        }
        if(LeftNode == -1 && RightNode != -1)//待删节点的左子树空，右子树非空
        {
            MemoryArray[DictGetIndexRight(PreNodeIndex)] = he(RightNode);
            MemoryArray[DictGetIndexPre(RightNode)] = he(PreNodeIndex);
        }
        if(LeftNode != -1 && RightNode != -1)
        {
            //寻找右侧子树中的第一个左侧节点
            int RightRoot = RightNode;
            int FindRightLeft = -1;
            while(true)
            {
                if(RightRoot==-1)break;
                else
                {
                    FindRightLeft = RightRoot;
                    RightRoot = MemoryArray[DictGetIndexLeft(RightRoot)].i();
                }
            }
            //把删掉节点的左孩子的父节点设为右侧最左节点
            MemoryArray[DictGetIndexPre(LeftNode)] = he(FindRightLeft);
            MemoryArray[DictGetIndexLeft(FindRightLeft)] = he(LeftNode);
            MemoryArray[DictGetIndexRight(PreNodeIndex)] = he(RightNode);
            MemoryArray[DictGetIndexPre(RightNode)] = he(PreNodeIndex);
        }
    }
    DictMemoryIndexManager.push(FindRes.MemoryIndex);
    MemoryArrayUsefulLength--;
}