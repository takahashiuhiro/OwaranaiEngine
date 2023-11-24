#pragma once
#include "../../CommonMathMoudle/Tensor.h"
#include "../../CommonDataStructure/Dict.h"
#include "OpsType.h"
#include "../ForwardFunction.h"
#include <cmath>

class ComputationalGraph;
class BaseOps
{
public:
    size_t OpsTypeName;
    Dict Params;
    ComputationalGraph* CG;
    std::string Nodeid;

    /**节点前的系数.*/
    using AddWeightType = std::map<std::string, float>;
    using AddWeightTypePtr = std::shared_ptr<AddWeightType>;
    /**节点是否转置.*/
    using TType = std::map<std::string, bool>;
    using TTypePtr = std::shared_ptr<TType>;
    /**节点选择的维度.*/
    using SelectDimType = std::map<std::string, size_t>;
    using SelectDimTypePtr = std::shared_ptr<SelectDimType>;
    using SelectDimsType = std::map<std::string, std::vector<size_t>>;
    using SelectDimsTypePtr = std::shared_ptr<SelectDimsType>;
    /**广播维度.*/
    using BroadCastToType = std::map<std::string, std::vector<size_t>>;
    using BroadCastToTypePtr = std::shared_ptr<BroadCastToType>;

    virtual ~BaseOps(){};
    /**前向计算.*/
    virtual void Forward() = 0;
    /**计算图的前向调用.*/
    void ForwardProcess();
    /**反向计算图设置.*/
    virtual void Backward();
    /**初始设置参数权重.*/
    virtual void ParamsDefinition();
    /**算子初始化.*/
    void CommonInit(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG);
    /**获取节点的输入节点列表.*/
    std::vector<std::string> &GetInputNodeList();
    std::vector<std::string> &GetInputNodeList(std::string InputNodeid);
    /**计算图标记后处理,只有在前向赋值计算以后才允许调用.*/
    void CGForwardProcess();
    /**.dropout后处理*/
    void CGForwardProcessDropout();

    virtual void AfterSettingShapeComputing() = 0;

    /**设置每个输入参数的常系数.*/
    void SetAddWeight(AddWeightType InputNodeWeight);
    /**获取输入参数的常系数.*/
    float GetAddWeight(std::string InputNodeid);
    /**设置每个输入参数是否为转置.*/
    void SetT(TType InputNodeIsT);
    /**获取输入参数是否为转置.*/
    bool GetT(std::string InputNodeid);
    /**设置该张量使用的输入维度.*/
    void SetSelectDim(SelectDimType InputNodeSelectDim);
    /**获取该张量使用的输入维度.*/
    size_t GetSelectDim(std::string InputNodeid);
    /**设置该张量使用的输入维度数组.*/
    void SetSelectDims(SelectDimsType InputNodeSelectDims);
    /**获取该张量使用的输入维度数组.*/
    std::vector<size_t> GetSelectDims(std::string InputNodeid);
    /**设置广播矩阵参数.*/
    void SetBroadCastTo(BroadCastToType BroadCastToShape);
    /**获取该张量使用的输入维度.*/
    std::vector<size_t> GetBroadCastTo(std::string InputNodeid);
    /**设置元素指数.*/
    void SetEleExponent(float Exponent);
    /**获取元素指数.*/
    float GetEleExponent();
    /**设置底数.*/
    void SetEleBaseNum(float BaseNum);
    /**获取底数.*/
    float GetEleBaseNum();
};
