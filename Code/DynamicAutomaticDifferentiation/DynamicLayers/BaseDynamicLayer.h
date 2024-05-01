#include "../DynamicTensor.h"

class BaseDynamicLayer
{
public:
	he Params;
	std::map<std::string, DynamicTensor>Weights;

	virtual void Init(he InputParams) = 0;
	virtual std::vector<DynamicTensor>Forward(std::vector<DynamicTensor>InputForwardList, he InputParams = he()) = 0;
};

