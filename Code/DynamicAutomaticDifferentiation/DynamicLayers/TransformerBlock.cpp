#include "TransformerBlock.h"
#include "Linear.h" 
#include "CausalSelfAttention.h"
#include "LayerNorm.h"

void MLPforTransformerBlock::SetLayerParams()
{
    NEmbd = Params["NEmbd"].i();
    Dropout = Params["Dropout"].f();
    Bias = Params["Bias"].i();
}

void MLPforTransformerBlock::InitContent()
{
    he CAttnParams = he::NewDict();
    CAttnParams["InFeatures"] = NEmbd;
    CAttnParams["OutFeatures"] = 4*NEmbd;
    CAttnParams["Bias"] = Bias;
    CreateNewLayer<Linear>("CFC", CAttnParams);
    he CProJParams = he::NewDict();
    CProJParams["InFeatures"] = 4*NEmbd;
    CProJParams["OutFeatures"] = NEmbd;
    CProJParams["Bias"] = Bias;
    CreateNewLayer<Linear>("CProj", CProJParams);
}

std::vector<DynamicTensor> MLPforTransformerBlock::Forward(std::vector<DynamicTensor>InputForwardList, he InputParams)
{
    auto X = SubLayers["CFC"]->Forward(InputForwardList)[0];
    X = X.GELU();
    X = SubLayers["CProj"]->Forward({X})[0];
    X = DynamicTensor::Dropout(X, Dropout);
    return {X};
}

void TransformerBlock::SetLayerParams()
{
    NHead = Params["NHead"].i();
    NEmbd = Params["NEmbd"].i();
    Dropout = Params["Dropout"].f();
    BlockSize = Params["BlockSize"].i();
    Bias = Params["Bias"].i();
}

void TransformerBlock::InitContent()
{
    he NormParam = he::NewDict();
    NormParam["NormalizedShape"] = he::NewList<int>({ NEmbd });
    NormParam["Bias"] = Bias;
    CreateNewLayer<LayerNorm>("LN1", NormParam);
    he CausalSelfAttentionParams = he::NewDict();
	CausalSelfAttentionParams["NHead"] = NHead;
	CausalSelfAttentionParams["NEmbd"] = NEmbd;
	CausalSelfAttentionParams["Dropout"] = Dropout;
	CausalSelfAttentionParams["BlockSize"] = BlockSize;
	CausalSelfAttentionParams["Bias"] = Bias;
    CreateNewLayer<CausalSelfAttention>("Attn", CausalSelfAttentionParams);
    CreateNewLayer<LayerNorm>("LN2", NormParam);
    he MLPforTransformerBlockParams = he::NewDict();
    MLPforTransformerBlockParams["NEmbd"] = NEmbd;
	MLPforTransformerBlockParams["Dropout"] = Dropout;
    MLPforTransformerBlockParams["Bias"] = Bias;
    CreateNewLayer<MLPforTransformerBlock>("MLP", MLPforTransformerBlockParams);
}

std::vector<DynamicTensor> TransformerBlock::Forward(std::vector<DynamicTensor>InputForwardList, he InputParams)
{
    auto X = InputForwardList[0];
    X = X + SubLayers["Attn"]->Forward(SubLayers["LN1"]->Forward(InputForwardList))[0];
    X = X + SubLayers["MLP"]->Forward(SubLayers["LN2"]->Forward({X}))[0];
    return {X};
}