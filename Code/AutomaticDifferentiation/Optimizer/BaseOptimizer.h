#pragma once
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "../../CommonMathMoudle/Tensor.h"
#include "../../CommonDataStructure/Dict.h"
#include "../ComputationalGraph.h"

class ComputationalGraph;

class BaseOptimizer
{
public:

    std::shared_ptr<ComputationalGraph> CG = nullptr;
    /**存张量的名称和对应的本体以及导数.*/
    std::map<std::string, std::pair<Tensor*, Tensor*>>TensorMap;
    /**计算完成后的位置.清除的时候直接删掉，因为所有权理应已经交了出去*/
    std::map<std::string, Tensor*>ResTensorMap;
    /**配置参数.*/
    Dict Params;

    /**从计算图中获取需要更新的计算节点.*/
    std::vector<std::string> GetWeightUpdateNodesByCG();
    /**在优化器里注册一个张量变量.*/
    void RegisterTensor(std::string TensorName);
    /**通过计算图注册所有张量.*/
    void RegisterAllTensorsByCG();
    /**对张量名称对应的数据进行赋值并处理.*/
    void AssignTensor(std::string TensorName,Tensor* ProtoTensor, Tensor* DTensor);
    /**从计算图同步张量指针.*/
    void SyncTensorByCG();
    /**把结果同步向计算图.*/
    void SyncTensorToCG();

    /**初始化优化器.*/
    virtual void Init();
    virtual void Init(std::shared_ptr<ComputationalGraph> InputCG);
    virtual void InitByCG(std::shared_ptr<ComputationalGraph> InputCG);

    /**如果需要在优化器里记录什么在这个函数里重载.*/
    virtual void InitTensorConfig(std::string TensorName){};
    /**需要处理赋值的张量重载这个函数.*/
    virtual void ProcessAssignTensor(std::string TensorName,Tensor* ProtoTensor, Tensor* DTensor){};
    /**通过梯度和预备数据等更新本值.*/
    virtual void Update() = 0;
    /**设置优化器默认参数.*/
    virtual void SetDefaultParams(){};
    /**清空结果数据.*/
    virtual void ClearData();
};