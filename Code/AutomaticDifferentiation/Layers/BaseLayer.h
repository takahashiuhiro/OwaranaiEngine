#pragma once
#include <iostream>
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "../../CommonDataStructure/CommonFuncHelpers.h"
#include "../../CommonMathMoudle/Tensor.h"
#include "../../CommonDataStructure/Dict.h"
#include "../ComputationalGraph.h"
#include "../ForwardFunction.h"

class BaseLayer
{
public:
    /**树上的链.*/
    std::string PreName = "";
    /**该层的名字.*/
    std::string LayerName = "";
    /**子节点们.*/
    std::map<std::string, std::shared_ptr<BaseLayer>>SubLayers;
    /**本层所属的权重矩阵.*/
    std::vector<std::string>WeightNodeArray;
    /**root的计算图，只允许root是非nullptr.*/
    std::shared_ptr<ComputationalGraph> CG = nullptr;
    /**父节点，root为nullptr.*/
    BaseLayer* ParentLayer = nullptr;
    /**设备数(同tensor的设备数).*/
    size_t DeviceNum = 0;
    std::vector<std::string> InputNodes;

    /**判断该层是否为root节点.*/
    bool IsRootNode();
    /**公共init，例如在构造函数的时候声明计算图等.调用这个函数前要先把该分给子层的分了，防止重复声明计算图，寄了.*/
    void CommonInit(BaseLayer* InputParentLayer, std::string InputLayerName, size_t ThisDeviceNum);
    /**在这里注册网络.*/
    void RegisterLayer(std::shared_ptr<BaseLayer>InputLayer);
    /**注册参数矩阵.*/
    void RegisterWeightNode(std::string InputNodeid,std::vector<size_t>InputTensorShape);
    /**注册输入矩阵.*/
    void RegisterInputNode(std::string InputNodeid,std::vector<size_t>InputTensorShape);
    /**注册常量矩阵.*/
    void RegisterConstNode(std::string InputNodeid,std::vector<size_t>InputTensorShape);
    /**根据层内相对名字获取绝对名字.*/
    std::string GetLayerNodeName(std::string InputNodeName);
    /**储存网络权重.*/
    void SaveToFile(std::string SavePath);
    /**加载网络权重.*/
    void LoadFromFile(std::string LoadPath);
    /**dfs的输出所有子层要保存权重的节点.*/
    std::vector<std::string> GetAllSubLayersNodeDfs();
    std::vector<std::string> GetAllSubLayersNodeDfs(bool AutoFlag);
    /**衔接树链.*/
    std::string GetLinkName(std::string PreStr, std::string NxtStr);

    /**进行前向构建的.*/
    virtual std::vector<std::string> Forward(std::vector<std::string>InputNodeArray){return {};};
    /**感觉应该还得有一个启动函数.*/
    virtual void Run(){};
    /**todo::缺一个能一次性冻结整个layer子树的函数.*/
};