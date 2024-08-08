#include "CommonFuncHelpers.h"

void print(bool Input) { std::cout << Input << std::endl; }
void print(int Input){std::cout << Input << std::endl;}
void print(std::string Input){std::cout << Input << std::endl;}
void print(float Input){std::cout << Input << std::endl;}
void print(double Input){std::cout << Input << std::endl;}
void print(const char* Input){std::cout << Input << std::endl;}
void print(size_t Input){std::cout << Input << std::endl;}
void print(char Input){std::cout << Input << std::endl;}

std::vector<std::string> LoadStringFromFile(std::string InputName)
{
    std::ifstream InputFile(InputName);
    if (!InputFile.is_open()) 
    {
        std::cerr << "Failed to open the file." << std::endl;
        return {};
    }
    std::vector<std::string> Res;
    std::string line;
    while (std::getline(InputFile, line)) Res.push_back(line);
    InputFile.close();
    return Res;
}

void SaveStringToFile(std::vector<std::string> StringVec,std::string InputName)
{
    std::ofstream InputFile(InputName);
    if (!InputFile.is_open()) 
    {
        std::cerr << "Failed to open the file." << std::endl;
        return;
    }
    for(auto&it:StringVec)InputFile.write(reinterpret_cast<const char*>(it.c_str()), it.length());
    InputFile.close();
}

std::vector<int> GenerateUniqueRandomNumbers(int Num, int Start, int End)
{
    int n = End - Start + 1;
    std::vector<int> Numbers(n);
    for(int a=0;a<n;a++)Numbers[a] = Start+a;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(Numbers.begin(), Numbers.end(), g);
    Numbers.resize(Num);
    return Numbers;
}