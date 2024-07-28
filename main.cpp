#include "Code/OEDynamic.h"

int main() 
{
    auto q = new GPT2Model();
    he Params = he::NewDict();
    Params["BlockSize"] = 1;
    Params["VocabSize"] = 1;
    Params["NLayers"] = 1;
    Params["NHead"] = 1;
    Params["NEmbd"] = 1;
    Params["Dropout"] = float(1.);
    Params["Bias"] = 1;
    q->Init(Params);
    print(q->GetNumParams());
}