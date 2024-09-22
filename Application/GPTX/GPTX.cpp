#include "GPTX.h"

GPTX::GPTX()
{
    TokenIdxTable = he::NewDict();
}

void GPTX::GenTokenIdxTable(std::string InputName)
{
    auto LoadData = LoadStringFromFile(InputName);
    std::map<std::string, int>Table;
    for(auto&it:LoadData)
    {
        for(int a=0;a+3<it.size();a+=3)
        {
            std::string TMPStr = it.substr(a,3);
            if(Table.find(TMPStr)==Table.end())Table[TMPStr] = 0;
            Table[TMPStr] += 1;
        }
    }
    std::vector<std::pair<std::string, int>>TableVec;
    for(auto&it:Table)TableVec.push_back(it);

    std::sort(TableVec.begin(), TableVec.end(), [](std::pair<std::string, int>& a,std::pair<std::string, int>& b) {return a.second > b.second;});

    for(int a=0;a<TableVec.size();a++)
    {
        TokenIdxTable[TableVec[a].first] = a+2;
        TokenIdxTable[a+2] = TableVec[a].first;
    }
    TokenIdxTable["\0"] = 0;
    TokenIdxTable[0] = "\0";
}

void GPTX::LoadTokenIdxTable(std::string InputName)
{
    TokenIdxTable = he::LoadFromString(LoadStringFromFile(InputName)[0]);
}

std::vector<int> GPTX::TextToVector(std::vector<std::vector<int>>&IndexVec,int BatchSize, int Length, std::vector<int>BatchVec)
{
    if(BatchVec.size()==0)
    {
        BatchVec = GenerateUniqueRandomNumbers(BatchSize, 0, IndexVec.size());
    }
    std::vector<int>Res;
    for(auto&it:BatchVec)
    {
        for(int a=0;a<Length;a++)
        {
            if(a < IndexVec[it].size())Res.push_back(IndexVec[it][a]);
            else Res.push_back(0);
        }
    }
    return Res;
}

std::vector<int> GPTX::StringMapToIndexByTokenIdxTable(std::string InputString)
{
    auto& it = InputString;
    std::vector<int>LastVec;
    for(int a=0;a<it.size();a+=3)
    {
        if(a==it.size()-1)
        {
            LastVec.push_back(0);
            continue;
        }
        std::string TMPStr = it.substr(a,3);
        if(TokenIdxTable.In(TMPStr))LastVec.push_back(TokenIdxTable[TMPStr].i());
        else LastVec.push_back(1);
    }
    return LastVec;
}

void GPTX::TrainConversation(std::string InputName)
{
    auto AllData = LoadStringFromFile(InputName);

    //划分训练和测试集
    auto TrainIndex = GenerateUniqueRandomNumbers(AllData.size()*0.375, 0, AllData.size()*0.5);
    std::vector<std::string>TrainSet, ValidSet;
    std::vector<int> Flag;
    for(int a=0;2*a+1<AllData.size();a++)Flag.push_back(1);
    for(int a=0;a<TrainIndex.size();a++)Flag[TrainIndex[a]] = 0;
    for(int a=0;a<Flag.size();a++)
    {
        std::string TMPStr = AllData[2*a].substr(0,AllData[2*a].size()-1) + AllData[2*a+1];
        if(!Flag[a])TrainSet.push_back(TMPStr);
        else ValidSet.push_back(TMPStr);
    }

    //通过词表转换
    std::vector<std::vector<int>>TrainSetIndex, ValidSetIndex;
    for(auto&it:TrainSet)TrainSetIndex.push_back(StringMapToIndexByTokenIdxTable(it));
    for(auto&it:ValidSet)ValidSetIndex.push_back(StringMapToIndexByTokenIdxTable(it));

    auto OptimizerIns = Optimizer::CreateSGD(LanguageModel->Parameters(), 0.001);

    int B = 10, T = 37;
    he TrainParams = he::NewDict();
    TrainParams["XShape"] = he::NewList<int>({B,T});

    float TrainHisLoss = 1e5, ValidHisLoss = 1e5;

    for(int Epoch=0;Epoch < 3000;Epoch++)
    {
        LanguageModel->Train();

        auto ThisEpochIndex = GenerateUniqueRandomNumbers(TrainSetIndex.size(), 0, TrainSetIndex.size()-1);
        int StartIndex = 0;
        float NewLoss = 0;

        while(StartIndex+B<ThisEpochIndex.size())
        {
            std::vector<int> BatchVec;
            int ThisB = 0;
            for(int a=StartIndex;a < std::min((int)ThisEpochIndex.size(), StartIndex+B); a++)
            {
                BatchVec.push_back(ThisEpochIndex[a]);
                ThisB++;
            }
            StartIndex += ThisB;
            auto TokenIndexVec = TextToVector(TrainSetIndex, ThisB, T, BatchVec);
            TrainParams["XData"] = he::NewList(TokenIndexVec);
            DynamicTensor X = LanguageModel->Forward({},TrainParams)[0];

            for(int a=0;a<TokenIndexVec.size();a++)
            {
                if(a%T==T-1)TokenIndexVec[a] = 0;
                else TokenIndexVec[a] = TokenIndexVec[a+1];
            }

            DynamicTensor Y = DynamicTensor::CreateOnehotTensor({ThisB,T}, TokenIndexVec, LanguageModel->Params["VocabSize"].i(), false, X.GetDeviceNum());
            auto LossRes = DynamicTensor::CrossEntropy(X,Y);
            NewLoss += LossRes.Ops->TensorPointer->GetV({0})*B;
            OptimizerIns.ZeroGrad();
            LossRes.Backward();
            OptimizerIns.Step();
        }

        NewLoss = NewLoss/ThisEpochIndex.size()/T;

        TrainHisLoss = std::min(TrainHisLoss, NewLoss);

        std::cout<<"train:Epoch: "<<Epoch<<"\t"<<"cur: "<<NewLoss<<"\t"<<"best: "<<TrainHisLoss<<std::endl;

        LanguageModel->Eval();

        ThisEpochIndex = GenerateUniqueRandomNumbers(ValidSetIndex.size(), 0, ValidSetIndex.size()-1);
        StartIndex = 0;
        NewLoss = 0;

        while(StartIndex+B<ThisEpochIndex.size())
        {
            std::vector<int> BatchVec;
            int ThisB = 0;
            for(int a=StartIndex;a < std::min((int)ThisEpochIndex.size(), StartIndex+B); a++)
            {
                BatchVec.push_back(ThisEpochIndex[a]);
                ThisB++;
            }
            StartIndex += ThisB;
            auto TokenIndexVec = TextToVector(ValidSetIndex, ThisB, T, BatchVec);
            TrainParams["XData"] = he::NewList(TokenIndexVec);
            DynamicTensor X = LanguageModel->Forward({},TrainParams)[0];

            for(int a=0;a<TokenIndexVec.size();a++)
            {
                if(a%T==T-1)TokenIndexVec[a] = 0;
                else TokenIndexVec[a] = TokenIndexVec[a+1];
            }

            DynamicTensor Y = DynamicTensor::CreateOnehotTensor({ThisB,T}, TokenIndexVec, LanguageModel->Params["VocabSize"].i(), false, X.GetDeviceNum());
            auto LossRes = DynamicTensor::CrossEntropy(X,Y);

            NewLoss += LossRes.Ops->TensorPointer->GetV({0})*B;
        }

        NewLoss = NewLoss/ThisEpochIndex.size()/T;

        if(ValidHisLoss > NewLoss)
        {
            ValidHisLoss = NewLoss;
            if(Epoch)LanguageModel->Save("../Application/GPTX/GPT2.weight.oe");
        }
        if(Epoch&&(Epoch%20==0))
        {
            std::string RootPath = "../Application/GPTX/GPT2_Epoch_";
            std::string EndPath = ".weight.oe";
            LanguageModel->Save(RootPath+NumberToString(Epoch)+EndPath);
        }

        std::cout<<"valid:Epoch: "<<Epoch<<"\t"<<"cur: "<<NewLoss<<"\t"<<"best: "<<ValidHisLoss<<std::endl;

    }
}

void GPTX::GenConversation(std::string InputSentense)
{
    LanguageModel->Eval();
    std::string ReturnStr = "";

    int T = 35;
    auto IndexVec = StringMapToIndexByTokenIdxTable(InputSentense);

    he ForwardParams = he::NewDict();

    for(int _=0;_<T;_++)
    {
        ForwardParams["XShape"] = he::NewList<int>({1,(int)IndexVec.size()});
        ForwardParams["XData"] = he::NewList(IndexVec);
        auto Res = LanguageModel->Forward({}, ForwardParams)[0].Copy();
        Res.Ops->TensorPointer->ToDevice(0);
        float RV = -1;
        int idx;
        for(size_t a=0;a<LanguageModel->Params["VocabSize"].i();a = a+1)
        {
            float GetV = Res.Ops->TensorPointer->GetV({0,IndexVec.size()-1,a});
            //print(GetV);
            if(RV < GetV)
            {
                RV = GetV;
                idx = a;
            }
        }
        //print(idx);
        IndexVec.push_back(idx);

        if(idx==0||idx==TokenIdxTable["."].i())break;
        ReturnStr += TokenIdxTable[idx].s();
    }

    ForwardParams["XShape"] = he::NewList<int>({1,(int)IndexVec.size()});
    ForwardParams["XData"] = he::NewList(IndexVec);
    auto X = LanguageModel->Forward({}, ForwardParams)[0].Copy();
    for(int a=0;a<IndexVec.size();a++)
    {
        if(a < IndexVec.size()-1)IndexVec[a] = IndexVec[a+1];
        else IndexVec[a]=0;
    }
    DynamicTensor Y = DynamicTensor::CreateOnehotTensor({1,(int)IndexVec.size()}, IndexVec, LanguageModel->Params["VocabSize"].i(), false, X.GetDeviceNum());
    //auto LossRes = DynamicTensor::CrossEntropy(X,Y);
    auto LossRes = (X-Y).Pow(2).Sum();
    //print(LossRes*(1./IndexVec.size()));
    print(InputSentense);
    print(ReturnStr);
}