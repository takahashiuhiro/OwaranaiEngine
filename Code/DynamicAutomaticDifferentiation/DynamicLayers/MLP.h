#pragma once
#include "BaseDynamicLayer.h"
#include "LayerNorm.h"

/*
*@Params
* InChannels  输入维度.
* HiddenChannels  隐藏层.
* Default:
* NormLayer = "None" 使用的norm种类.
* ActivationLayer = "ReLU" 激活函数, None不用.
* Bias = true 偏置.
* Dropout = 0 给dropout用的百分比.
.*/

class MLP : public BaseDynamicLayer
{

};