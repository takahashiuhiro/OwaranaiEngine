#include <iostream>
#include <string>
#include <functional>
#include <vector>
#include <queue>
#include "Log.h"
#include "CommonFuncHelpers.h"

struct heType
{
    static const size_t NONE = 0;
    static const size_t INT = 1;
    static const size_t STRING = 2;
    static const size_t FLOAT = 3;
    static const size_t LIST = 4;
    static const size_t DICT = 5;

    template<typename T = size_t>
    static std::string ToString(size_t InputType)
    {
        if(InputType == NONE)return "None";
        if(InputType == INT)return "INT";
        if(InputType == STRING)return "STRING";
        if(InputType == FLOAT)return "FLOAT";
        if(InputType == LIST)return "LIST";
        if(InputType == DICT)return "DICT";
    }
};

struct DictFindDfsInfo
{
    /**
    存在时:
    HasResult = 1;
    MemoryIndex = 内存块下标;
    不存在时:
    HasResult = 0;
    MemoryIndex = 内存块下标的父节点;
    IsLeft = 左0/右1;
    没有节点的时候:
    HasResult = 0;
    MemoryIndex = -1;
    IsLeft = -1;
    .*/
    int HasResult = 0;//是否存在
    int MemoryIndex = -1;//如果存在，内存块下标的值，如果不存在，作为输出未知位置的父节点，-1代表没有父节点
    int IsLeft = -1;//-1代表不知道，0是左，1是右
};

struct he
{
    /**基础类型值.*/
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

    /**基础类型读取.*/
    int i();
    std::string s();
    float f();

    /**基础类型写入.*/
    void w(int Input);
    void w(std::string Input);
    void w(float Input);
    void w(double Input);

    /**公共函数.*/
    std::string DumpToString();

    ~he();

    void Set(he* ThisOther,he Other);

    he operator + (he Other)const;
    he operator - (he Other)const;
    he operator * (he Other)const;
    he operator * (int Other)const;
    he operator * (double Other)const;
    he operator * (float Other)const;
    he operator / (he Other)const;
    he operator = (he Other)const;
    he& operator = (he Other);
    bool operator == (he Other)const;
    he& operator [] (int Other);
    he& operator [] (he Other);
    bool operator < (he Other)const;
    bool operator <= (he Other)const;
    bool operator > (he Other)const;
    bool operator >= (he Other)const;

    /**容器内存管理.*/
    std::vector<he>MemoryArray;
    int MemoryArrayUsefulLength = 0;

    /**容器公共部分.*/
    he size();
    std::hash<int> HashInt;
    std::hash<std::string> HashString;
    std::hash<float> HashFloat;
    int hash(he Other);

    /**list类型管理.*/
    static he NewList(int InputLength = 0);
    void append(he Other);

    /**dict类型管理.*/
    std::priority_queue<int,std::vector<int>, std::greater<int> >DictMemoryIndexManager;
    int SplayRoot = -1;
    static he NewDict();
    void DictApplyNewBlock(int BlockNum = 1);//申请一个新的dict块大小的内存
    int DictGetNextIndex();//得到下一个要插入的内存块
    void DictSetMemoryBlockByIndex(int InputIndex, he InputKey, he InputValue, bool IsInit = 0);//在内存块index上插入一对pair
    int DictGetIndexKey(int InputIndex);//查找已知内存块的属性位置，下同
    int DictGetIndexValue(int InputIndex);
    int DictGetIndexLeft(int InputIndex);
    int DictGetIndexRight(int InputIndex);
    int DictGetIndexPre(int InputIndex);
    he& DictFromKtoV(he InputKey);
    DictFindDfsInfo SplayFind(he InputKey);//在splay上查询一个key在哪个内存块上，如果查不到返回-1
    DictFindDfsInfo SplayInputFindDfs(int RootIndex, int InputKey);//具体的dfs查询
    int SplayInsert(he InputKey, he InputValue);//在splay上插入一对k-v,返回内存块
    void SplayPrintForDebugArray();//debug用
    void SplayPrintForDebugTree();
    void SplayPrintForDebugDfs(int Root);
    void SplayPrintForDebugSingleNode(int Root);
    void Splay(int InputIndex);//通过内存块的index把输入的index的节点rotate到根上去
    void SplayDelete(he InputKey);//通过key删除节点
};