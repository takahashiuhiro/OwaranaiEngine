#include <memory>
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
#include <iostream>
#include "Code/OEDynamic.h"


int main()
{
    Embedding* q = new Embedding();
    he params = he::NewDict();
    params["NumEmbeddings"] = 4;
    params["EmbeddingDim"] = 3;
    params["PaddingIdx"] = 1;
    q->Init(params);
    he fp = he::NewDict();
    fp["BatchSize"] = 2;
    fp["TextureLength"] = 3;
    fp["XData"] = he::NewList<int>({0,1,2,0,2,3});
    //print(q->Weights["Weight"]);
    //print(q->Forward({}, fp));
    auto gg = q->Forward({}, fp);
    auto ss = (gg[0]-1).Pow(2).Sum();
    ss.Backward();
    print(q->Weights["Weight"].Grad());
}
