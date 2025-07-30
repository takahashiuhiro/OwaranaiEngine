#pragma once
#include "BaseBlackBoxOptimizer.h"

/**
 * 分块高斯分布
 */
struct BlockGaussian
{
    DynamicTensor Mean;// 均值.
    DynamicTensor Var, VarL, VarInv, VarLDet, VarLInv; // 协方差，协方差的LU分解, 协方差的逆, 协方差的行列式

    bool IsInit = false; //是否初始化完成.

    void Init(DynamicTensor InputMean, DynamicTensor InputVar)
    {
        Mean = InputMean;
        Var = InputVar;
        VarL = InputVar.Cholesky();
        VarInv = InputVar.Inverse();
        VarLDet = VarL.Det_Symmetric(VarL);
        VarLInv = VarL.Inverse();
        IsInit = true;
    }
};

struct GMM
{
    /**这个vector里每一个元素都是独立的高斯分布, 是一个大高斯的分块矩阵. */
    std::vector<BlockGaussian> PartialBlock;
    DynamicTensor PartialRate;// 不同高斯的系数.

    void Init(std::vector<DynamicTensor>InputMeans, std::vector<DynamicTensor> InputVars, DynamicTensor InputPartialRate)
    {
        for(size_t a = 0; a< InputMeans.size();a++)
        {
            PartialBlock.push_back(BlockGaussian());
            PartialBlock.back().Init(InputMeans[a], InputVars[a]);
        }
        PartialRate = InputPartialRate;
    }

    /**按照分块矩阵采样 */
    std::vector<DynamicTensor> Sample(std::vector<int> InputVec, int Seed = -1)
    {
        std::vector<DynamicTensor> res;
        int DeviceNum = PartialRate.GetDeviceNum();
        for(size_t a  = 0;a < PartialBlock.size();a++)
        {
            int BlockDim = PartialBlock[a].Mean.Shape().back();
            res.push_back(DynamicTensor::SampleFromOtherGaussian(BlockDim, InputVec, PartialBlock[a].Mean, PartialBlock[a].Var,  PartialBlock[a].VarL, Seed, DeviceNum));
        }
        return res;
    }

    /**
     * 给出一个只有分块部分的采样求概率密度
     * @param InputPartialSample 样本
     * @param InputIdx 第几块
     */
    DynamicTensor GetSingleBlockPDF(DynamicTensor InputPartialSample, int InputIdx)
    {
        // 未测试
        auto& ThisBlock = PartialBlock[InputIdx];
        return DynamicTensor::ProbabilityDensity_Gaussian(InputPartialSample, ThisBlock.Mean, ThisBlock.VarInv, ThisBlock.VarLDet);
    }

};

struct GMMwithSample
{
    GMM Distribution;
    std::vector<DynamicTensor> PartialSample;
    DynamicTensor EvalRes;

    void Init(GMM InputGMM, std::vector<DynamicTensor> InputSample, DynamicTensor InputEvalRes)
    {
        Distribution = InputGMM;
        PartialSample = InputSample;
        EvalRes = InputEvalRes;
    }
};

struct GMMHistory
{

    int HistoryLength;

    std::deque<GMMwithSample> GMMContent;

    void Add(GMMwithSample InputGMMContent)
    {
        GMMContent.push_back(InputGMMContent);
        while(GMMContent.size() > HistoryLength)
        {
            GMMContent.pop_front();
        }
    }

    /**
     * 通过Eval的结果评测整个窗口里每个样例的影响
     * @param Beta 把每个Eval评测结果先进行归一化, 然后再乘以beta进行softmax
     */
    DynamicTensor GetSampleEvalRate(double Beta)
    {
        std::vector<DynamicTensor> Res;
        for(auto&it:GMMContent)Res.push_back(it.EvalRes);
        DynamicTensor CatEval = DynamicTensor::Cat(Res, 0); // (SampleNum, CosmosNum, 1)
        DynamicTensor CatMeanEval = CatEval.Mean({0}, true);// (1, CosmosNum, 1)
        DynamicTensor CatVarEval = CatEval.Var({0}, true);// (1, CosmosNum, 1)
        DynamicTensor NormEval = (CatEval - CatMeanEval)*(CatVarEval + 1e-6).Pow(-0.5);// (SampleNum, CosmosNum, 1)
        DynamicTensor FinalRes = (NormEval*Beta).Softmax(0);// (SampleNum, CosmosNum, 1)
        return FinalRes;
    }

    /**
     * 获取一个blcok里所有历史结果的融合结果
     */
    DynamicTensor GetAllSample(int BlockIndex)
    {
        std::vector<DynamicTensor> TMPSample;
        for(auto&it:GMMContent)TMPSample.push_back(it.PartialSample[BlockIndex]);
        return DynamicTensor::Cat(TMPSample);
    }

    /**
     * 计算当前block的所有采样，当前概率密度处以混合历史概率密度修正
     * @param AllSample (HistoryLength, CosmosNum, Dim)
     */
    DynamicTensor GetAllSampleMeanPDF(DynamicTensor AllSample, int BlockIndex)
    {
        DynamicTensor Res;
        for(size_t a = 0;a < GMMContent.size()-1;a++)
        {
            auto& ThisGaussian = GMMContent[a].Distribution.PartialBlock[BlockIndex];
            if(!a)Res = DynamicTensor::ProbabilityDensity_Gaussian(AllSample, ThisGaussian.Mean, ThisGaussian.VarInv, ThisGaussian.VarLDet);
            else Res = Res + DynamicTensor::ProbabilityDensity_Gaussian(AllSample, ThisGaussian.Mean, ThisGaussian.VarInv, ThisGaussian.VarLDet);
        }
        // 当前这一个也是要当成分子的，避免重复计算
        auto& CurGaussian = GMMContent[GMMContent.size()-1].Distribution.PartialBlock[BlockIndex];
        DynamicTensor CurPD = DynamicTensor::ProbabilityDensity_Gaussian(AllSample, CurGaussian.Mean, CurGaussian.VarInv, CurGaussian.VarLDet);
        if(GMMContent.size() > 1)Res = Res + CurPD;
        else Res = CurPD;
        return CurPD*Res.Pow(-1.)*(GMMContent.size()*1.);
    }

};


template<typename TargetType>
struct NESGMMBased: public BaseBlackBoxOptimizer<TargetType>
{

    int DimNum; //优化变量维度
    int SampleNum; //采样数
    int CosmosNum; //GMM的高斯数
    int HistoryLength; //采样使用得历史窗口.
    int MaxItNum; // 最大的迭代次数
    double LearingRate_Mean; //学习率
    double LearingRate_Var; //学习率
    double EvalScoreInitRate; // 把零阶cost映射到对梯度效应的时候的初始值
    double EvalScoreMaxRate; // 把零阶cost映射到对梯度效应的时候的最大值

    GMM TargetDistribution; // 目标分布

    GMMHistory SampleSelector; // 从这里选择最好的样本以及他们对应的分布

    bool IsWarmStart = false; // 是不是已经热启动了，如果热启动了就可以跳过随机初始化

    virtual void ParamsInit()
    {
        DimNum = this->Params["DimNum"].i();
        SampleNum = this->Params.DictGet("SampleNum", 30).i();
        CosmosNum = this->Params.DictGet("CosmosNum", 1).i();
        HistoryLength = this->Params.DictGet("HistoryLength", 1).i();
        MaxItNum = this->Params.DictGet("MaxItNum", 1).i();
        LearingRate_Mean = this->Params.DictGet("LearingRate_Mean", 0.1).f();
        LearingRate_Var = this->Params.DictGet("LearingRate_Var", 0.02).f();
        EvalScoreInitRate = this->Params.DictGet("EvalScoreInitRate", 0.1).f();
        EvalScoreMaxRate = this->Params.DictGet("EvalScoreMaxRate", 0.8).f();
    }

    /**
     * 初始化目标分布，一般来说都应该给出warm start，如果没给的话就从标准正态分布开始了，这种情况跑得慢是一定的
     */
    void DistributionInit()
    {
        if(IsWarmStart)return;
        // 这里只有debug的时候是这样的，正式版这里应该权衡一下吧，怎么并行比较好，因为独立的高斯是串行的，同一个高斯是并行的
        auto Mean = DynamicTensor({(size_t)CosmosNum, (size_t)DimNum}, false, this->DeviceNum);
        Mean.Fill(0);
        auto Var = DynamicTensor::CreateUnitTensor({CosmosNum, DimNum, DimNum}, false, this->DeviceNum);
        auto PartialRate = DynamicTensor({(size_t)CosmosNum}, false, this->DeviceNum);
        PartialRate.Fill(1./CosmosNum);
        TargetDistribution.Init({Mean}, {Var}, PartialRate);
    }

    /**
     * 对于历史窗口的初始化
     */
    void HistoryInit()
    {
        SampleSelector.HistoryLength = HistoryLength;
    }

    virtual DynamicTensor Solve()
    {        
        DistributionInit();
        HistoryInit();

        for(size_t ItIdx = 0; ItIdx < MaxItNum; ItIdx++)
        {
            // 把历史已经完成的更新采样加入历史
            auto Add2History = [this]()
            {
                GMMwithSample CurrentGMM;
                std::vector<DynamicTensor> SampleRes = TargetDistribution.Sample({SampleNum});
                DynamicTensor CurrentEvalRes = this->TargetObj->Eval(DynamicTensor::Cat(SampleRes, SampleRes[0].Shape().size()-1));
                CurrentGMM.Init(TargetDistribution, SampleRes, CurrentEvalRes);
                SampleSelector.Add(CurrentGMM);
            };

            // 根据迭代轮数来调f(x)的影响
            auto GetBeta = [&ItIdx, this]()
            {
                return EvalScoreInitRate* (MaxItNum - ItIdx*1.)/MaxItNum + EvalScoreMaxRate*(ItIdx*1.)/MaxItNum;
            };
            
            // 把Eval的结果的修正，这个无需按照分块来确认
            auto GetF = [&GetBeta, this]()
            {
                double Beta = GetBeta();
                DynamicTensor F = SampleSelector.GetSampleEvalRate(Beta);//(SampleNum, CosmosNum, 1), 对于每个采样而言的直接效用，不考虑采样偏移
                return F.View({SampleNum, CosmosNum});
            };

            auto GetDeltaMean = [this](DynamicTensor& AllSample, int BlockIndex)
            {
                auto& ThisBlock = TargetDistribution.PartialBlock[BlockIndex];
                auto TargetVarInv = ThisBlock.VarInv.View({1, CosmosNum, DimNum, DimNum});
                auto TargetZeroBiasSample = (AllSample - ThisBlock.Mean).View({SampleNum,CosmosNum,DimNum,1}); //(SampleNum,CosmosNum,Dim)
                DynamicTensor Res = TargetVarInv%TargetZeroBiasSample;
                return Res.View({SampleNum,CosmosNum,DimNum});
            };

            auto GetDeltaVar = [this](DynamicTensor& AllSample, int BlockIndex)
            {
                auto& ThisBlock = TargetDistribution.PartialBlock[BlockIndex];
                DynamicTensor Q = ThisBlock.VarLInv;//todo



            };

            // 按照不同的分块信息更新目标分布里的每个分块高斯
            auto UpdateBlockWeight = [this,&GetF,&GetDeltaMean,&GetDeltaVar](int BlockIndex)
            {
                auto& ThisBlock = TargetDistribution.PartialBlock[BlockIndex];
                // 得到所有要用的样例
                DynamicTensor AllSample = SampleSelector.GetAllSample(BlockIndex);
                // 计算所有窗口中的样例每个分块高斯在历史的平均密度系数
                DynamicTensor FinalF = GetF()*SampleSelector.GetAllSampleMeanPDF(AllSample, BlockIndex);
                //DynamicTensor Delta_Mean = ThisBlock.VarInv; // (CosmosNum,Dim,Dim)
                DynamicTensor DeltaMean = (GetDeltaMean(AllSample, BlockIndex)*FinalF.View({SampleNum,CosmosNum,1})).Mean({0});//(CosmosNum,Dim)

                GetDeltaVar(AllSample, BlockIndex);

                //print(DeltaMean.Shape()); //(SampleNum,CosmosNum,Dim)
            };

            // 根据前一帧内容采样，把历史已经完成的更新采样加入历史
            Add2History();
            // 获取修改采样飘逸以后的更新F(todo)
            for(int BlockIndex = 0;BlockIndex < TargetDistribution.PartialBlock.size();BlockIndex++)UpdateBlockWeight(BlockIndex);
            
        }
        // 挑选历史样本
        // 计算各种重要性
        // 计算梯度



        return DynamicTensor(); //暂时返回
    }
    
};
