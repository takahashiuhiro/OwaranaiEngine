#pragma once
#include "../DynamicTensor.h"

/*
*@Params
* Default
* DeviceNum = 0 CPU\GPU.
.*/

class BaseDynamicLayer
{
public:
	he Params;
	size_t DeviceNum = 0;
	bool IsEval = false;
	std::map<std::string, DynamicTensor>Weights;
	std::map<std::string, std::shared_ptr<BaseDynamicLayer>>SubLayers;
	/**存一些不会被加到参数里的常量. */
	std::map<std::string, DynamicTensor>Buffers;

	template<typename T>
	void CreateNewLayer(std::string LayerName,he InputParams)
	{
		SubLayers[LayerName] = std::make_shared<T>();
		SubLayers[LayerName]->Init(InputParams);
	}

	std::vector<DynamicTensor> Parameters();
	std::map<std::string, DynamicTensor> StateDict();
	void StateDictDFS(std::map<std::string, DynamicTensor>& ResMp, std::string PreStr);
	void Eval();
	void Train();
	void SetCommonDefaultParams();
	void SetParams(he InputParams);
	void Init(he InputParams);

	virtual void SetLayerParams() = 0;
	virtual void InitContent() = 0;
	virtual std::vector<DynamicTensor> Forward(std::vector<DynamicTensor>InputForwardList, he InputParams = he()) = 0;
};

