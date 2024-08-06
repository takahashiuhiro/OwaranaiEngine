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
    for(auto&it:TrainSet)
    {
        TrainSetIndex.push_back({});
        auto& LastVec = TrainSetIndex[TrainSetIndex.size()-1];
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
    }
    for(auto&it:ValidSet)
    {
        ValidSetIndex.push_back({});
        auto& LastVec = ValidSetIndex[ValidSetIndex.size()-1];
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
    }

    auto OptimizerIns = Optimizer::CreateSGD(LanguageModel->Parameters(), 0.0001);

    int B = 10, T = 70;
    he TrainParams = he::NewDict();
    TrainParams["XShape"] = he::NewList<int>({B,T});

    double HisLoss = 1e5;

    for(int Epoch=0;Epoch < 1500;Epoch++)
    {
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
            DynamicTensor Y = DynamicTensor::CreateOnehotTensor({ThisB,T}, TokenIndexVec, LanguageModel->Params["VocabSize"].i(), false, X.GetDeviceNum());
            auto LossRes = DynamicTensor::CrossEntropy(X,Y);

            NewLoss += LossRes.Ops->TensorPointer->GetV({0})*B;
            OptimizerIns.ZeroGrad();
            LossRes.Backward();
            OptimizerIns.Step();
            //print(LossRes);
        }

        NewLoss = NewLoss/ThisEpochIndex.size();

        if(HisLoss > NewLoss)
        {
            HisLoss = NewLoss;
            if(Epoch)LanguageModel->Save("../Application/GPTX/GPT2.weight.oe");
        }

        std::cout<<"Epoch: "<<Epoch<<"\t"<<"cur: "<<NewLoss<<"\t"<<"best: "<<HisLoss<<std::endl;

    }
}