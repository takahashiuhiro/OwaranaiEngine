#include "Code/OEDynamic.h"
#include "Application/GPTX/GPTX.h"

int main() 
{
    he Params = he::NewDict();
    Params["BlockSize"] = 1;
    Params["VocabSize"] = 1;
    Params["NLayers"] = 1;
    Params["NHead"] = 1;
    Params["NEmbd"] = 1;
    Params["Dropout"] = float(1.);
    Params["Bias"] = 1;

    GPTX a;
    a.Init<GPT2Model>(Params);
    //a.LanguageModel->Load("gg.oeh");
    //print(a.LanguageModel->Parameters());

    a.GenTokenIdxTable("../DataSet/1.烤鸭场景描述.txt");

}