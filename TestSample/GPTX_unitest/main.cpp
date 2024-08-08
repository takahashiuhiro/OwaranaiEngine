#include "Code/OEDynamic.h"
#include "Application/GPTX/GPTX.h"

int main() 
{
    he Params = he::NewDict();
    Params["BlockSize"] = 500;
    Params["VocabSize"] = 200;
    Params["NLayers"] = 1;
    Params["NHead"] = 32;
    Params["NEmbd"] = 1536;
    Params["Dropout"] = float(0.5);
    Params["Bias"] = 1;
    Params["DeviceNum"] = 1;

    GPTX a;
    a.Init<GPT2Model>(Params);

    std::cout<<"参数量: "<<a.LanguageModel->GetNumParams()<<" m"<<std::endl;

    //生成词表
    //a.GenTokenIdxTable("../DataSet/pkduck/pkduck.data.oe");
    //储存词表
    //SaveStringToFile({a.TokenIdxTable.DumpToString()}, "../DataSet/pkduck/pkduck.table.oe");
    //加载词表
    a.LoadTokenIdxTable("../DataSet/pkduck/pkduck.table.oe");
    //加载权重
    a.LanguageModel->Load("../Application/GPTX/test_res/GPT2_l1_nh32_ne1536.weight.oe");
    //训练
    //a.TrainConversation("../DataSet/pkduck/pkduck.data.oe");
    //生成
    a.GenConversation("他们的烤鸭香味也很浓郁。");
}