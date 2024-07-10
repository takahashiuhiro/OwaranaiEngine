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
    DynamicTensor X = InputForwardList[0];
    Log::Assert(X.Shape().size()==3, "Size of shape of inputs of CausalSelfAttention must be 3");
    int B = X.Shape()[0];
    int T = X.Shape()[1];
    int C = X.Shape()[2];
    auto QKV = SubLayers["CAttn"]->Forward({X})[0].Split(NEmbd,2);
    DynamicTensor Q = QKV[0].View({B,T,NHead,C/NHead}).Transpose(1,2);
    DynamicTensor K = QKV[1].View({B,T,NHead,C/NHead}).Transpose(1,2);
    DynamicTensor V = QKV[2].View({B,T,NHead,C/NHead}).Transpose(1,2);
    DynamicTensor Att = (Q%K.Transpose(-2,-1))*(1./std::sqrt(K.Shape()[K.Shape().size()-1]));
    
    return {};
}