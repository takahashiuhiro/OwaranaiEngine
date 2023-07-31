#pragma once
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <any>
#include <variant>
#include <typeinfo>

class Dict {
public:
    using KeyType = std::variant<int, std::string, size_t>;

    template <typename T>
    void Set(KeyType key, T value) 
    {
        if(typeid(key)==typeid(int)||typeid(key)==typeid(size_t))
        {
            data_[std::get<0>(key)] = value;
            return;
        }
        else
        {
            data_[std::get<1>(key)] = value;
            return;
        }
        data_[key] = value;
    }

    template <typename T>
    T Get(KeyType key)
    {
        if(typeid(key)==typeid(int)||typeid(key)==typeid(size_t))
        {
            return std::any_cast<T>(data_[std::get<0>(key)]);
        }
        else
        {
            return std::any_cast<T>(data_[std::get<1>(key)]);
        }
    }

private:
    std::map<KeyType, std::any> data_;
};