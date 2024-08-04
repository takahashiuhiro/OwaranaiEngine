#include "CommonFuncHelpers.h"

void print(bool Input) { std::cout << Input << std::endl; }
void print(int Input){std::cout << Input << std::endl;}
void print(std::string Input){std::cout << Input << std::endl;}
void print(float Input){std::cout << Input << std::endl;}
void print(double Input){std::cout << Input << std::endl;}
void print(const char* Input){std::cout << Input << std::endl;}
void print(size_t Input){std::cout << Input << std::endl;}
void print(char Input){std::cout << Input << std::endl;}

std::vector<std::string> LoadTxtFromFile(std::string InputName)
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