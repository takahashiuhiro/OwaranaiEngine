#include "../DynamicTensor.h"

class BaseDynamicLayer
{
public:
	he Params;
	size_t DeviceNum = 0;
	std::map<std::string, DynamicTensor>Weights;
	std::map<std::string, std::shared_ptr<BaseDynamicLayer>>SubLayers;

	template<typename T>
	void CreateNewLayer(std::string LayerName,he InputParams)
	{
		SubLayers[LayerName] = std::make_shared<T>();
		SubLayers[LayerName]->Init(InputParams);
	}

	std::vector<DynamicTensor> Parameters();
	std::map<std::string, DynamicTensor> StateDict();
	void StateDictDFS(std::map<std::string, DynamicTensor>& ResMp, std::string PreStr);

	virtual void Init(he InputParams) = 0;
	virtual std::vector<DynamicTensor> Forward(std::vector<DynamicTensor>InputForwardList, he InputParams = he()) = 0;
};

