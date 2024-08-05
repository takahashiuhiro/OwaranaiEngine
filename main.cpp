#include "Code/OEDynamic.h"
#include "Application/GPTX/GPTX.h"

int main() 
{
    he Params = he::NewDict();
    Params["BlockSize"] = 500;
    Params["VocabSize"] = 508;
    Params["NLayers"] = 1;
    Params["NHead"] = 1;
    Params["NEmbd"] = 32;
    Params["Dropout"] = float(0.5);
    Params["Bias"] = 1;

    GPTX a;
    a.Init<GPT2Model>(Params);

    print(a.LanguageModel->GetNumParams());

    a.LoadTokenIdxTable("../DataSet/pkduck/pkduck.table.oe");

    a.TrainConversation("../DataSet/pkduck/pkduck.data.oe");
}