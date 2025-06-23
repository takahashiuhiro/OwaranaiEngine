#pragma once
#include "BaseBlackBoxOptimizer.h"

template<typename TargetType>
struct NESGMMBased: public BaseBlackBoxOptimizer<TargetType>
{

    int DimNum; //优化变量维度
    int SampleNum; //采样数
    int CosmosNum; //粒子数
    double Beta; //softmax的系数
    double LearingRate_Var; //学习率
    double LearingRate_Mean; //学习率
    double Gamma; // 用来调斥力的


    virtual void ParamsInit()
    {
        DimNum = this->Params["DimNum"].i();
        SampleNum = this->Params.DictGet("SampleNum", 30).i();
        CosmosNum = this->Params.DictGet("CosmosNum", 30).i();
        Beta = this->Params.DictGet("Beta", 0.3).f();
        LearingRate_Mean = this->Params.DictGet("LearingRate_Mean", 0.05).f();
        LearingRate_Var = this->Params.DictGet("LearingRate_Var", 0.01).f();
        Gamma = this->Params.DictGet("Gamma", 1.).f();
    }

    virtual DynamicTensor Solve()
    {        
        std::vector<DynamicTensor>MeanSet, VarSet;
        std::vector<int> SampleNumSet;
        int SampleSumAll = SampleNum;
        for(int a=0;a<CosmosNum;a++)
        {
            MeanSet.push_back(DynamicTensor({(size_t)DimNum}, 0, this->DeviceNum));
            MeanSet.back().FillRandomValNormal(); // 随机填充均值
            //MeanSet.back().Fill(0);
            VarSet.push_back(DynamicTensor({(size_t)DimNum}, 0, this->DeviceNum));
            //VarSet.back().FillRandomValNormal();
            //VarSet.back() = VarSet.back().Abs(); // 随机填充方差
            VarSet.back().Fill(10);
            SampleNumSet.push_back(int(SampleNum/CosmosNum) + int(a < (SampleNum%CosmosNum)));
        }

        auto GetSampleRes = [&SampleNumSet, &MeanSet, &VarSet, this]()
        {
            std::vector<DynamicTensor>SampleSet;
            for(int a = 0;a < CosmosNum;a++)
            {
                SampleSet.push_back(DynamicTensor::SampleFromMulGaussian(MeanSet[a], VarSet[a], {SampleNumSet[a]}, false, this->DeviceNum));
            }
            return DynamicTensor::Cat(SampleSet);
        };

        for(int a = 0;a < 50;a++)
        {
            DynamicTensor SampleRes = GetSampleRes();
            //print(SampleRes);
            DynamicTensor EvalRes = this->TargetObj->Eval(SampleRes)*(-1); // 这里是cost，越烂的越低，所以取负的 
            //print(EvalRes);
            DynamicTensor W = (EvalRes * (Beta)).Softmax(0);
            //print(W);
            auto GetR = [&MeanSet, &VarSet, &SampleRes, this]()
            {
                std::vector<DynamicTensor> PSingle;
                for(int a=0;a<CosmosNum;a++)
                {
                    PSingle.push_back(SampleRes.GetProbabilityDensityFromGaussian(MeanSet[a], VarSet[a]).View({1, -1}));
                }
                DynamicTensor MergeAllPSingle = DynamicTensor::Cat(PSingle).Transpose(0, 1)*(1./CosmosNum); // 行: 样本， 列: 高斯
                DynamicTensor MergeResSum = MergeAllPSingle.Sum({1}, true).Pow(-1.);
                return MergeAllPSingle*MergeResSum;
            };
            DynamicTensor R = GetR();
            //print(R);

            auto GetMergeParamsTheta = [this](auto& MergeVector)
            {
                // 矩阵化处理，为了并行
                std::vector<DynamicTensor>InputVec;
                for(auto& it:MergeVector)InputVec.push_back(it.View({1,-1,DimNum}));// shape(采样数, 粒子数, 样本维度)
                return DynamicTensor::Cat(InputVec, 1);
            };

            DynamicTensor MergeMean = GetMergeParamsTheta(MeanSet);
            DynamicTensor MergeVar = GetMergeParamsTheta(VarSet);


            auto GetDeltaMean = [&W, &R, &MergeMean, &MergeVar, &SampleRes, this]()
            {
                DynamicTensor XView = SampleRes.View({-1, 1, DimNum});
                DynamicTensor XZeroMean = XView - MergeMean;
                DynamicTensor VarInv = MergeVar.Pow(-1);
                DynamicTensor VarInvProdXZeroMean = VarInv*XZeroMean;
                DynamicTensor WR_View = (W*R).View({SampleNum, -1, 1});
                DynamicTensor JMean = (WR_View*VarInvProdXZeroMean).Sum({0});
                return JMean;
            };

            auto JMean = GetDeltaMean();

            auto GetDeltaVar = [&W, &R, &MergeMean, &MergeVar, &SampleRes, this]()
            {
                DynamicTensor VarInv_2_Nega = MergeVar.Pow(-1)*(-1);
                DynamicTensor VarInv_4 = MergeVar.Pow(-2);
                DynamicTensor XView = SampleRes.View({-1, 1, DimNum});
                DynamicTensor XZeroMean_2 = (XView - MergeMean).Pow(2);
                DynamicTensor InnerV = (XZeroMean_2*VarInv_4 + VarInv_2_Nega)*0.5;
                DynamicTensor WR_View = (W*R).View({SampleNum, -1, 1});
                DynamicTensor JVar = (WR_View*InnerV).Sum({0});
                return JVar;
            };

            auto JVar = GetDeltaVar();

            auto GetFRepel = [&MergeMean, &MergeVar, this]()
            {
                /* ① 带宽 h²：用平均方差并乘 Gamma */
                DynamicTensor h2   = MergeVar.Mean({});          // scalar
                DynamicTensor invh = (h2 + 1e-12).Pow(-1);       // 1/h²

                /* ② Δμ_{kl} */
                DynamicTensor mu_k = MergeMean.View({-1,1,DimNum});
                DynamicTensor mu_l = MergeMean.View({1,-1,DimNum});
                DynamicTensor dmu  = mu_k - mu_l;                // (K,K,D)

                /* ③ 高斯核 */
                DynamicTensor dist2  = dmu.Pow(2).Sum({2}, true);        // (K,K,1)
                DynamicTensor kernel = (dist2 * (-0.5) * invh).Eleexp(M_E); // (K,K,1) 0-1

                /* ④ 斥力向量 */
                DynamicTensor repel = (kernel * dmu * invh).Sum({1});    // (K,D)
                return repel * (1./CosmosNum);                        // 均匀平均
            };

            //auto GetFRepel = [&MergeMean, &MergeVar, this]()
            //{
            //    auto GetH = [&MergeVar, this]()
            //    {
            //        DynamicTensor HRes = MergeVar.Sum({},true);
            //        double HResNum = Gamma/CosmosNum;
            //        return HRes.View({1,1,-1})*HResNum;
            //    };
            //    DynamicTensor H = GetH();
            //    DynamicTensor DualMergeMean_1 = MergeMean.View({-1,1,DimNum});
            //    DynamicTensor DualMergeMean_2 = MergeMean.View({1,-1,DimNum});
            //    DynamicTensor DualMergeMean = DualMergeMean_1 - DualMergeMean_2;
            //    DynamicTensor Dis = DualMergeMean.Pow(2.).Sum({2}, true);
            //    DynamicTensor NormDis = H.Pow(-1)*(-0.5);
            //    DynamicTensor CosmosKernel = (Dis*NormDis).Eleexp(M_E);
            //    DynamicTensor DuelDis = CosmosKernel*DualMergeMean;
            //    DynamicTensor Res = DuelDis.Sum({1})*(1./CosmosNum);
            //    return Res.Sum({0});
            //};

            auto GetMeanUpdateDelta = [&JMean, &GetFRepel, &a, this]()
            {
                double FRepelDeacy = 0.08;
                DynamicTensor FRepel = GetFRepel();
                DynamicTensor FRepel_Deacy = FRepel * std::max(1-a*FRepelDeacy, 0.) * Gamma;
                DynamicTensor Res = JMean - FRepel_Deacy;
                return Res * LearingRate_Mean;
            };

            DynamicTensor MergeMeanUpdateDelta = GetMeanUpdateDelta();

            auto GetVarLogUpdateDelta = [&MergeVar, &JVar, this]()
            {
                DynamicTensor ProtoRes = MergeVar *JVar* LearingRate_Var;
                DynamicTensor ProtoCopy = ProtoRes.Copy();
                ProtoCopy.Fill(3.);
                DynamicTensor ClipRes = (ProtoRes - ProtoCopy).ReLU();
                print(ProtoRes);
                print(ProtoCopy);
                print(ClipRes);
                return ProtoRes - ClipRes;
            };

            DynamicTensor MergeVarLogUpdateDelta = GetVarLogUpdateDelta();

            DynamicTensor NextMergeMean = MergeMean + MergeMeanUpdateDelta;
            DynamicTensor NextMergeVar = MergeVar * (MergeVarLogUpdateDelta).Eleexp(M_E);

            std::cout<<"--mean--::start::"<<a<<std::endl;
            print(MergeMean);
            print("--mean--::2");
            print(NextMergeMean);
            print("--mean--::end");
            MeanSet = NextMergeMean.Split(1, 1);
            for(auto& EachMean:MeanSet)EachMean = EachMean.View({-1});
            print("--var--::start");
            print(MergeVar);
            print("--var--::2");
            print(NextMergeVar);
            print("--var--::3");
            VarSet = NextMergeVar.Split(1, 1);
            for(auto& EachVar:VarSet)EachVar = EachVar.View({-1});

            print("");
        }
        print("all_end");
        //todo 迭代采样eval啥的..
        return DynamicTensor(); //todo::暂代返回值
    }
    
};
