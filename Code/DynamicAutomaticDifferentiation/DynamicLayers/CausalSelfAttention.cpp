#include "CausalSelfAttention.h"
#include "Linear.h"

void CausalSelfAttention::SetLayerParams()
{
    NHead = Params["NHead"].i();
    NEmbd = Params["NEmbd"].i();
    Dropout = Params["Dropout"].f();
    BlockSize = Params["BlockSize"].i();
    Bias = Params["Bias"].i();
}

void CausalSelfAttention::InitContent()
{
    he CAttnParams = he::NewDict();
    CAttnParams["InFeatures"] = NEmbd;
    CAttnParams["OutFeatures"] = 3*NEmbd;
    CAttnParams["Bias"] = Bias;
    CreateNewLayer<Linear>("CAttn", CAttnParams);
    he CProJParams = he::NewDict();
    CProJParams["InFeatures"] = 3*NEmbd;
    CProJParams["OutFeatures"] = NEmbd;
    CProJParams["Bias"] = Bias;
    CreateNewLayer<Linear>("CProj", CProJParams);
    auto BiasTensor = DynamicTensor({1,1,BlockSize*1U, BlockSize*1U},0,DeviceNum);
    BiasTensor.Fill(1.);
    Buffers["Bias"] = BiasTensor.Tril();
}

std::vector<DynamicTensor> CausalSelfAttention::Forward(std::vector<DynamicTensor>InputForwardList, he InputParams)
{
    //todo
    return {};
}