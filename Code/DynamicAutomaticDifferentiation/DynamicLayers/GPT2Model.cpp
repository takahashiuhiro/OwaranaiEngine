#include "GPT2Model.h"

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
    
}

std::vector<DynamicTensor> GPT2Model::Forward(std::vector<DynamicTensor>InputForwardList, he InputParams)
{

}