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
		InputParams["DeviceNum"] = int(DeviceNum);
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
	void Save(std::string InputName);
	void Load(std::string InputName);

	//递归在所有的子layer中执行一个函数，被执行的函数第一个参数必须是this
	template <typename Func, typename... Args>
	void Apply(Func FuncIns, Args&&... ArgsIns) 
	{
		auto ArgsTuple = std::make_tuple(std::forward<Args>(ArgsIns)...);
		std::apply(FuncIns, ArgsTuple);
		auto CurPointer = std::get<0>(ArgsTuple);
		for(auto&SubLayerPair:CurPointer->SubLayers)
		{
			ArgsTuple = ReplaceElement<0>(ArgsTuple, SubLayerPair.second.get());
			std::apply([this, FuncIns](auto&&... UnpackedArgs) {
                this->Apply(FuncIns, std::forward<decltype(UnpackedArgs)>(UnpackedArgs)...);
            }, ArgsTuple);
		}
	}

	virtual void SetLayerParams() = 0;
	virtual void InitContent() = 0;
	virtual std::vector<DynamicTensor> Forward(std::vector<DynamicTensor>InputForwardList, he InputParams = he()) = 0;

	float GetNumParams();
};

