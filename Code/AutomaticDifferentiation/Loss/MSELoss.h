#include "BaseLoss.h"

class MSELoss:public BaseLoss
{
public:
    virtual void Build(std::vector<std::string>InputCGNodeList, std::vector<std::string>LabelNodeList);
};