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

    //根据数据生成loss
    DynamicTensor GetLoss();

};