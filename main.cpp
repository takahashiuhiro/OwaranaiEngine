#include "Code/OEDynamic.h"
#include "Application/GPTX/GPTX.h"

int main() 
{

    size_t pp = 512;
    DynamicTensor q({pp,pp}, 0, 1);
    DynamicTensor w({pp,pp}, 0, 1);
    q.Fill(1);
    w.Fill(2);
    auto rr = (q%w).Sum();

    print(rr);

    return 0;
    he Params = he::NewDict();
    Params["BlockSize"] = 500;
    Params["VocabSize"] = 508;
    Params["NLayers"] = 2;
    Params["NHead"] = 2;
    Params["NEmbd"] = 2;
    Params["Dropout"] = float(0.5);
    Params["Bias"] = 1;
    Params["DeviceNum"] = 0;

    GPTX a;
    a.Init<GPT2Model>(Params);

    print(a.LanguageModel->GetNumParams());

    a.LoadTokenIdxTable("../DataSet/pkduck/pkduck.table.oe");

    a.TrainConversation("../DataSet/pkduck/pkduck.data.oe");
}