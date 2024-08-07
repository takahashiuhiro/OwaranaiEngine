#include "GPT2Model.h"
#include "Embedding.h"
#include "TransformerBlock.h"
#include "Linear.h"
#include "LayerNorm.h"

void GPT2Model::SetLayerParams()
{
    BlockSize = Params["BlockSize"].i();
    VocabSize = Params["VocabSize"].i();
    NLayers = Params["NLayers"].i();
    NHead = Params["NHead"].i();
    NEmbd = Params["NEmbd"].i();
    Dropout = Params["Dropout"].f();
    Bias = Params["Bias"].i();
}

void GPT2Model::InitContent()
{
    he WTEParams = he::NewDict();
    WTEParams["NumEmbeddings"] = VocabSize;
    WTEParams["EmbeddingDim"] = NEmbd;
    CreateNewLayer<Embedding>("WTE", WTEParams);
    he WPEParams = he::NewDict();
    WPEParams["NumEmbeddings"] = BlockSize;
    WPEParams["EmbeddingDim"] = NEmbd;
    CreateNewLayer<Embedding>("WPE", WPEParams);
    he TransformerBlockParams = he::NewDict();
    TransformerBlockParams["NHead"] = NHead;
    TransformerBlockParams["NEmbd"] = NEmbd;
    TransformerBlockParams["Dropout"] = Dropout;
    TransformerBlockParams["BlockSize"] = BlockSize;
    TransformerBlockParams["Bias"] = Bias;
    for(int a=0;a < NLayers;a++)
    {
        std::string TMPLayerName = std::string("TransformerBlock_") + NumberToString(a);
        CreateNewLayer<TransformerBlock>(TMPLayerName, TransformerBlockParams);
        TransformerBlockLayerNames.push_back(TMPLayerName);
    }
    he LNFParams = he::NewDict();
    LNFParams["NormalizedShape"] = he::NewList<int>({NEmbd});
    LNFParams["Bias"] = Bias;
    CreateNewLayer<LayerNorm>("LNF", LNFParams);
    he LMHeadParams = he::NewDict();
    LMHeadParams["InFeatures"] = NEmbd;
    LMHeadParams["OutFeatures"] = VocabSize;
    LMHeadParams["Bias"] = 0;
    CreateNewLayer<Linear>("LMHead", LMHeadParams);
    SubLayers["WTE"]->Weights["Weight"] = SubLayers["LMHead"]->Weights["Weight"];//这一句原文自己也不知道能不能跑不行就删了
    Apply(InitWeights, static_cast<BaseDynamicLayer*>(this));
    Apply(InitWeightsCProj, static_cast<BaseDynamicLayer*>(this), NLayers);
}

std::vector<DynamicTensor> GPT2Model::Forward(std::vector<DynamicTensor>InputForwardList, he InputParams)
{
    int B = InputParams["XShape"][0].i();
    int T = InputParams["XShape"][1].i();
    Log::Assert(T <= BlockSize, "Block Size is Not Enough For Length.");

    he TokEmbParams = he::NewDict();
    TokEmbParams["XShape"] = he::NewList<int>({B, T});
    TokEmbParams["XData"] = InputParams["XData"];
    DynamicTensor TokEmb = SubLayers["WTE"]->Forward({}, TokEmbParams)[0];

    he PosEmbParams = he::NewDict();
    PosEmbParams["XShape"] = he::NewList<int>({T});
    PosEmbParams["XData"] = he::NewList(MathArange(0, T, 1));
    DynamicTensor PosEmb = SubLayers["WPE"]->Forward({}, PosEmbParams)[0];

    auto X = DynamicTensor::Dropout(TokEmb + PosEmb, Dropout);
    for(auto&TransformerBlockLayerName:TransformerBlockLayerNames)
    {
        X = SubLayers[TransformerBlockLayerName]->Forward({X})[0];
    }
    X = SubLayers["LNF"]->Forward({X})[0];
    X = SubLayers["LMHead"]->Forward({X})[0];
    return {X};
}

void GPT2Model::InitWeights(BaseDynamicLayer* CurLayer)
{
    if(typeid(*CurLayer) == typeid(Linear))
    {
        CurLayer->Weights["Weight"].FillRandomValNormal(0, 0.02);
        if(CurLayer->Weights.find("Bias")!=CurLayer->Weights.end())
        {
            CurLayer->Weights["Bias"].Fill(0);
        }
    }
    else if(typeid(*CurLayer) == typeid(Embedding))
    {
        CurLayer->Weights["Weight"].FillRandomValNormal(0, 0.02);
    }
}

void GPT2Model::InitWeightsCProj(BaseDynamicLayer* CurLayer, int ConfigNLayers)
{
    for(auto&SubLayerPair:CurLayer->SubLayers)
    {
        if(SubLayerPair.first == "CProj")
        {
            SubLayerPair.second->Weights["Weight"].FillRandomValNormal(0, 0.2/std::sqrt(2*ConfigNLayers));
        }
    }
}