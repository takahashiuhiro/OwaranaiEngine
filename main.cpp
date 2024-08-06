#include "Code/OEDynamic.h"
#include "Application/GPTX/GPTX.h"

int main() 
{

    //size_t pp = 2500;
    //DynamicTensor q({pp,pp}, 0, 0);
    //DynamicTensor w({pp,pp}, 0, 0);
    //q.Fill(1);
    //w.Fill(2);
    //auto rr = (q%w).Sum();
    //print(rr);
    //return 0;
    he Params = he::NewDict();
    Params["BlockSize"] = 500;
    Params["VocabSize"] = 508;
    Params["NLayers"] = 2;
    Params["NHead"] = 4;
    Params["NEmbd"] = 256;
    Params["Dropout"] = float(0.5);
    Params["Bias"] = 1;
    Params["DeviceNum"] = 1;

    GPTX a;
    a.Init<GPT2Model>(Params);

    //print(a.LanguageModel->GetNumParams());
    std::cout<<"参数量: "<<a.LanguageModel->GetNumParams()<<" m"<<std::endl;

    a.LoadTokenIdxTable("../DataSet/pkduck/pkduck.table.oe");
    //a.LanguageModel->Load("../Application/GPTX/GPT2.weight.oe");
    a.TrainConversation("../DataSet/pkduck/pkduck.data.oe");
}