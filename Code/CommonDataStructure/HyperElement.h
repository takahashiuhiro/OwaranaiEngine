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
    static he NewDict();
    void DictApplyNewBlock(int BlockNum = 1);//申请一个新的dict块大小的内存
    int DictGetNextIndex();//得到下一个要插入的内存块
    void DictSetMemoryBlockByIndex(int InputIndex, he InputKey, he InputValue);//在内存块index上插入一对pair
    int DictGetIndexKey(int InputIndex);//查找已知内存块的属性位置，下同
    int DictGetIndexValue(int InputIndex);
    int DictGetIndexLeft(int InputIndex);
    int DictGetIndexRight(int InputIndex);
    int SplayFind(he InputKey);//在splay上查询一个key在哪个内存块上，如果查不到返回-1
    int SplayInputFindDfs(int RootIndex, int InputKey);//具体的dfs
    void SplayInsert(he InputKey, he InputValue);//在splay上插入一对k-v
};