#include "../../Code/OEDynamic.h"

struct GPTX
{
    //语言模型
    std::shared_ptr<BaseDynamicLayer> LanguageModel;
    //查的token表, 正反都在
    he TokenIdxTable;

    GPTX();

    //初始化模型
    template<typename T>
    void Init(he ModelParams)
    {
        LanguageModel = std::make_shared<T>();
        LanguageModel->Init(ModelParams);
    }

    //通过输入数据构造查询表,0是停止，1是未知
    void GenTokenIdxTable(std::string InputName);
    //加载已经生成好的词表
    void LoadTokenIdxTable(std::string InputName);
    //把文本通过查词表的方式转换成数组
    std::vector<int> StringMapToIndexByTokenIdxTable(std::string InputString);
    //把文本转换成指定batch，长度的一维vector
    std::vector<int> TextToVector(std::vector<std::vector<int>>&IndexVec,int BatchSize = 10, int Length = 70, std::vector<int>BatchVec = {});

    //训练对话
    void TrainConversation(std::string InputName);
    //生成对话
    void GenConversation(std::string InputSentense);

};