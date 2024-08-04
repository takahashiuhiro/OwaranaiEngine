#include "../../Code/OEDynamic.h"

struct GPTX
{
    std::shared_ptr<BaseDynamicLayer> LanguageModel;

    GPTX();

    template<typename T>
    void Init(he ModelParams)
    {
        LanguageModel = std::make_shared<T>();
        LanguageModel->Init(ModelParams);
    }

    //DynamicTensor GetLoss()

};