#include "Embedding.h"
#include "Linear.h"

void Embedding::SetLayerParams()
{
    NumEmbeddings = Params["NumEmbeddings"].i();
    EmbeddingDim = Params["EmbeddingDim"].i();
    if (Params.In("PaddingIdx"))
    {
        PaddingIdx = Params["PaddingIdx"].i();
        PaddingIdxSwitch = true;
    }
	else 
    {
        PaddingIdx = -1e8;
        PaddingIdxSwitch = false;
    }
    if (Params.In("MaxNorm"))
    {
        MaxNorm = Params["MaxNorm"].f();
        MaxNormSwitch = true;
    }
	else 
    {
        MaxNorm = -1e9;
        MaxNormSwitch = false;
    }
    if (Params.In("NormType"))NormType = Params["NormType"].f();
	else NormType = 2.;
    if (Params.In("ScaleGradByFreq"))ScaleGradByFreq = Params["ScaleGradByFreq"].i();
	else ScaleGradByFreq = false;
    if (Params.In("Sparse"))Sparse = Params["Sparse"].i();
	else Sparse = false;
}

void Embedding::InitContent()
{
    Weights["Weight"] = DynamicTensor({(size_t)NumEmbeddings, (size_t)EmbeddingDim}, true, DeviceNum);
    Weights["Weight"].FillRandomValNormal();
    if(PaddingIdxSwitch)
    {
        DynamicTensor AllOneTensor = DynamicTensor({1,(size_t)EmbeddingDim},0,DeviceNum);
        AllOneTensor.Fill(1.);
        Buffers["VecAllOneConst"] = (DynamicTensor::CreateOnehotTensor({1}, {PaddingIdx}, NumEmbeddings, 0, DeviceNum).Transpose(-1,-2)*AllOneTensor)*(-1) + 1;
    }
}

std::vector<DynamicTensor> Embedding::Forward(std::vector<DynamicTensor>InputForwardList, he InputParams)
{
    std::vector<int>XShape;
    InputParams["XShape"].v(XShape);
    std::vector<int>XData;
    InputParams["XData"].v(XData);
    DynamicTensor X = DynamicTensor::CreateOnehotTensor(XShape, XData, NumEmbeddings, true, DeviceNum);
    if(PaddingIdxSwitch)return {X%(Weights["Weight"]*Buffers["VecAllOneConst"])};
    return {X%Weights["Weight"]};
}