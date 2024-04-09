#pragma once
#include "BaseLayer.h"
class PositionalEncodingLayer :public BaseLayer
{
public:
    PositionalEncodingLayer() {};
    PositionalEncodingLayer(BaseLayer* ParentThis, std::string ThisLayerName, size_t ThisDeviceNum, size_t DModel, size_t MaxLen);

    virtual std::vector<std::string> Forward(std::vector<std::string>InputNodeArray);

    size_t DModel;
    size_t MaxLen;
};