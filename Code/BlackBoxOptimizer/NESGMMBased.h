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
    DynamicTensor GetSampleEvalRate(int BlockIndex, double Beta)
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
     * 返回前beta的样例以及评测结果
     */
    std::pair<DynamicTensor, DynamicTensor> SampleEvalRankRate(int BlockIndex, double Beta)
    {
        int SingleTimeSampleNum = GMMContent[0].PartialSample[BlockIndex].Shape()[0];
        int CosmosNum = GMMContent[0].PartialSample[BlockIndex].Shape()[1];
        int DimNum = GMMContent[0].PartialSample[BlockIndex].Shape()[2];
        int ContentNum = GMMContent.size();
        int MaxSampleNum = ContentNum*SingleTimeSampleNum;
        int ResMaxSampleNum = MaxSampleNum*Beta;
        int DeviceNum =  GMMContent[0].PartialSample[BlockIndex].GetDeviceNum();
        
        auto SamplePairCMP = [](auto& Input_1, auto& Input_2){return Input_1.second > Input_2.second;};
        
        std::vector<std::priority_queue<std::pair<std::vector<double>, double>,std::vector<std::pair<std::vector<double>, double>>,decltype(SamplePairCMP)>> SampleQueue;
        for (int i = 0; i < CosmosNum; ++i) SampleQueue.emplace_back(SamplePairCMP);
        for(auto&it:GMMContent)
        {
            auto& ThisSample = it.PartialSample[BlockIndex];
            auto& ThisEvalRes = it.EvalRes;

            int ThisSampleNum = ThisSample.Shape()[0];

            auto SampleContent = ThisSample.Ops->TensorPointer->GetDevicePointer();
            auto EvalContent = ThisEvalRes.Ops->TensorPointer->GetDevicePointer();

            for(int CosmosIdx = 0;CosmosIdx < CosmosNum;CosmosIdx++)
            {
                for(int CurSampleIdx = 0;CurSampleIdx < ThisSampleNum;CurSampleIdx++)
                {
                    int ThisVecIdx = CurSampleIdx*CosmosNum + CosmosIdx;
                    std::vector<double>ThisSampleV;
                    for(int DimIndex = 0;DimIndex < DimNum;DimIndex++)
                    {
                        ThisSampleV.push_back(SampleContent[ThisVecIdx*DimNum + DimIndex]);
                    }
                    SampleQueue[CosmosIdx].emplace(std::move(ThisSampleV), EvalContent[ThisVecIdx]);

                    while(SampleQueue[CosmosIdx].size()>ResMaxSampleNum)SampleQueue[CosmosIdx].pop();
                }
            }
        }

        std::vector<float>SampleResContent;
        std::vector<float>EvalResContent;

        for(int a = 0;a < ResMaxSampleNum;a++)
        {
            for(int b = 0;b < CosmosNum;b++)
            {
                for(int c = 0;c < DimNum; c++)
                {
                    SampleResContent.push_back(SampleQueue[b].top().first[c]);
                }
                SampleQueue[b].pop();
                EvalResContent.push_back(std::log(1 + std::max(0.,(a+1)*1.0/MaxSampleNum)));
            }
        }

        DynamicTensor ResSample = DynamicTensor({(size_t)ResMaxSampleNum, (size_t)CosmosNum, (size_t)DimNum}, SampleResContent, false ,DeviceNum);
        DynamicTensor ResEval = DynamicTensor({(size_t)ResMaxSampleNum, (size_t)CosmosNum, 1}, EvalResContent, false ,DeviceNum);
        return std::make_pair(ResSample,ResEval);
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
    DynamicTensor GetAllSampleMeanPDF(DynamicTensor AllSample, int BlockIndex, DynamicTensor& CurPD)
    {
        std::vector<DynamicTensor>AllLogPDF;
        for(size_t a = 0;a < GMMContent.size();a++)
        {
            auto& ThisGaussian = GMMContent[a].Distribution.PartialBlock[BlockIndex];
            DynamicTensor CurLogPDF = DynamicTensor::ProbabilityDensity_Log_Gaussian(AllSample, ThisGaussian.Mean, ThisGaussian.VarInv, ThisGaussian.VarLDet);
            AllLogPDF.push_back(CurLogPDF);
            if(a + 1 == GMMContent.size())
            {
                CurPD = CurLogPDF;
            }
        }

        DynamicTensor MaxAlpha = DynamicTensor::Cat(AllLogPDF).Max({0},true);

        DynamicTensor Res;
        for(size_t a = 0;a < AllLogPDF.size();a++)
        {
            if(!a) Res = (AllLogPDF[a] - MaxAlpha).Eleexp(M_E);
            else Res = Res + (AllLogPDF[a] - MaxAlpha).Eleexp(M_E);
        }
        return CurPD - Res.EleLog() - MaxAlpha + std::log(AllLogPDF.size());//通过所有的log pdf求pdf的和的log
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
    double LearingRate_B;//学习率
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
        HistoryLength = this->Params.DictGet("HistoryLength", 2).i();
        MaxItNum = this->Params.DictGet("MaxItNum", 5000).i();
        LearingRate_Mean = this->Params.DictGet("LearingRate_Mean", 0.1).f();
        LearingRate_Var = this->Params.DictGet("LearingRate_Var", 0.02).f();
        LearingRate_B = this->Params.DictGet("LearingRate_B", 0.02).f();
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

            auto GetDeltaMean = [this](DynamicTensor& AllSample, int BlockIndex)
            {
                auto& ThisBlock = TargetDistribution.PartialBlock[BlockIndex];
                return AllSample - ThisBlock.Mean;
                //auto TargetVarInv = ThisBlock.VarInv.View({1, CosmosNum, DimNum, DimNum});
                //auto TargetZeroBiasSample = (AllSample - ThisBlock.Mean).View({-1,CosmosNum,DimNum,1}); //(SampleNum,CosmosNum,Dim)
                //DynamicTensor Res = TargetVarInv%TargetZeroBiasSample;
                //print((AllSample - ThisBlock.Mean).Shape());
                //return Res.View({-1,CosmosNum,DimNum});
            };

            auto GetDeltaVar = [this](DynamicTensor& AllSample, int BlockIndex)
            {
                // 理论上这种shape操作在我们的框架里这么操作只有在自动微分关闭的时候才能做，不然会炸
                auto& ThisBlock = TargetDistribution.PartialBlock[BlockIndex];
                DynamicTensor VarLInvT = ThisBlock.VarLInv.Transpose(-1,-2);//todo
                DynamicTensor XMinusMean = AllSample - ThisBlock.Mean;
                XMinusMean.Shape().push_back(1);
                DynamicTensor Q = VarLInvT%XMinusMean;
                DynamicTensor Q_T = Q.Transpose(-1,-2);
                DynamicTensor MatrixQQ = Q%Q_T;
                DynamicTensor UnitTensor = DynamicTensor::CreateUnitTensor(MatrixQQ.ShapeInt(), 0, this->DeviceNum);
                DynamicTensor Res = (MatrixQQ*UnitTensor - UnitTensor)*0.5;
                return Res;
            };

            // 按照不同的分块信息更新目标分布里的每个分块高斯
            auto UpdateBlockWeight = [this,&GetDeltaMean,&GetDeltaVar](int BlockIndex)
            {
                auto& ThisBlock = TargetDistribution.PartialBlock[BlockIndex];
                DynamicTensor AllSampleCurLogPDF;
                // 得到筛选后的样例以及他们的效用
                auto RankSampleResPair = SampleSelector.SampleEvalRankRate(BlockIndex, 0.5);
                // 得到所有要用的样例
                DynamicTensor AllSample = RankSampleResPair.first;
                DynamicTensor F = RankSampleResPair.second.View({-1, CosmosNum});

                // 计算所有窗口中的样例每个分块高斯在历史的平均密度系数，顺便返回他的概率密度后面要用
                DynamicTensor FinalF = F*SampleSelector.GetAllSampleMeanPDF(AllSample, BlockIndex, AllSampleCurLogPDF).Eleexp(M_E);
                //计算对均值的导数
                DynamicTensor DeltaMean = (GetDeltaMean(AllSample, BlockIndex)*FinalF.View({-1,CosmosNum,1})).Mean({0});//(CosmosNum,Dim)
                //计算对斜方差矩阵重参数化后中间的对角阵的导数，这个要乘过去
                DynamicTensor DeltaVar = (GetDeltaVar(AllSample, BlockIndex)*FinalF.View({-1,CosmosNum,1,1})).Mean({0});//(CosmosNum,Dim,Dim)
                DynamicTensor NewMean = ThisBlock.Mean + DeltaMean*LearingRate_Mean;
                DynamicTensor UnitVarShape = DynamicTensor::CreateUnitTensor(DeltaVar.ShapeInt(), false, this->DeviceNum);
                DynamicTensor NewVar = ThisBlock.VarL%((DeltaVar*LearingRate_Var).Eleexp(M_E)*UnitVarShape)%ThisBlock.VarL.Transpose(-1,-2);//(CosmosNum,Dim,Dim)
                ThisBlock.Init(NewMean, NewVar);
                return AllSampleCurLogPDF;
            };
            
            // 根据前一帧内容采样，把历史已经完成的更新采样加入历史
            Add2History();
            // 获取修改采样漂移以后的更新F(todo)
            DynamicTensor AllSampleLogPDF; //所有block的高斯密度的log
            for(int BlockIndex = 0;BlockIndex < TargetDistribution.PartialBlock.size();BlockIndex++)
            {
                auto tmp_res = UpdateBlockWeight(BlockIndex);
                if(!BlockIndex)AllSampleLogPDF = tmp_res;
                else AllSampleLogPDF = AllSampleLogPDF+tmp_res;
            }
            DynamicTensor AllSampleMean = AllSampleLogPDF.Mean({0});
            TargetDistribution.PartialRate = (TargetDistribution.PartialRate + AllSampleMean.Softmax(0)*LearingRate_B).Softmax(0);
            
            //debug--
            print("mean----------");
            print(TargetDistribution.PartialBlock[0].Mean);
            DynamicTensor gg = DynamicTensor({(size_t)CosmosNum,1,(size_t)DimNum},0);
            gg.Fill(1);
            print("varr----------");
            print(gg%TargetDistribution.PartialBlock[0].Var);

            //print(TargetDistribution.PartialBlock[0].Var);
        }
        return DynamicTensor(); //暂时返回
    }
};
