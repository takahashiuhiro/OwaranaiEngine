#pragma once
#include <iostream>
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "../../CommonMathMoudle/Tensor.h"
#include "../../CommonDataStructure/Dict.h"
#include "../ComputationalGraph.h"
#include "../Ops/OpsType.h"

class BaseLayer
{
public:
    /**树上的链.*/
    std::string PreName = "";
    /**该层的名字.*/
    std::string LayerName = "";
    /**子节点们.*/
    std::map<std::string, std::shared_ptr<BaseLayer>>SubLayers;//todo::别忘了这里要释放内存
    /**root的计算图，只允许root是非nullptr.*/
    std::shared_ptr<ComputationalGraph> CG = nullptr;
    /**父节点，root为nullptr.*/
    BaseLayer* ParentLayer = nullptr;
    /**设备数(同tensor的设备数)*/
    size_t DeviceNum = 0;

    /**判断该层是否为root节点.*/
    bool IsRootNode();
    /**公共init，例如在构造函数的时候声明计算图等.调用这个函数前要先把该分给子层的分了，防止重复声明计算图，寄了*/
    void CommonInit(BaseLayer* InputParentLayer, std::string InputLayerName, size_t ThisDeviceNum);
    /**在这里注册网络.*/
    void RegisterLayer(std::shared_ptr<BaseLayer>InputLayer);
    /**注册参数矩阵.*/
    void RegisterWeightNode();
    /**公共析构需要调用的.*/
    void CommonDestroy();

    /**进行前向构建的.*/
    virtual void Forward(){};
    /**感觉应该还得有一个启动函数.*/
    virtual void Run(){};
};