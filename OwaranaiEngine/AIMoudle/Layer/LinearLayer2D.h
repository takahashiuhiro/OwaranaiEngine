#pragma once
#include "BaseLayer.h"

/**一个简单的线性分类器*/

struct LinearLayer2D: BaseLayer
{
public:
    /**
     * Params:
     * "InputShape":vector<size_t>
     * "OutputShape":vector<size_t>
     * "BatchSize":vector<size_t>
    */
    virtual void ForwardBuild();
};