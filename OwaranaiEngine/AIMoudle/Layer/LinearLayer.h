#pragma once
#include "BaseLayer.h"

/**一个简单的线性分类器*/

struct LinearLayer: BaseLayer
{
public:
    virtual void ForwardBuild();
};