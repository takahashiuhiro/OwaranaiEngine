#include "BaseOptimizer.h"

class SGDOptimizer:public BaseOptimizer
{
public:
    virtual void SetDefaultParams();
    virtual void UpdateContent();
};