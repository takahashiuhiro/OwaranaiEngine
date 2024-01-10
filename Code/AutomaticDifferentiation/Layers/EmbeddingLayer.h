#pragma once
#include "BaseLayer.h"

class EmbeddingLayer:public BaseLayer
{
public:
    EmbeddingLayer(){};
    EmbeddingLayer(BaseLayer* ParentThis,std::string ThisLayerName, size_t ThisDeviceNum, size_t NumEmbeddings, size_t EmbeddingDim, std::pair<bool, size_t> PaddingIdx={false, 0}, bool Freeze = false,std::pair<bool, float> MaxNorm={false, 0}, float NormType=2.0, bool ScaleGradByFreq=false, bool Sparse=false);

    virtual std::vector<std::string> Forward(std::vector<std::string>InputNodeArray);

    /**储存构造信息.*/
    size_t NumEmbeddings;
    size_t EmbeddingDim;
    std::pair<bool, size_t> PaddingIdx;
    bool Freeze;
    std::pair<bool, float> MaxNorm;
    float NormType;
    bool ScaleGradByFreq;
    bool Sparse;
    /**嵌入权重变量的名字.*/
    std::string WeightNode;
    /**嵌入权重与padding.*/
    std::string PaddingWeightNode;
    /**first记录shape，second是data.*/
    std::vector<std::pair<std::vector<size_t>,std::vector<size_t>>>EmbeddingChangeList;

    /**输入待转化的数据.*/
    void AddEmbeddingNode(std::vector<size_t> InputShape, std::vector<size_t> InputData);

    /**加载预训练权重.*/
    void FromPretrained(Tensor* PretrainedTensor);
};