#include "ui.hpp"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <locale>

// this is copied from https://stackoverflow.com/a/217605/5272201
static inline void ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
                return !std::isspace(ch);
            }));
}
static inline void rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
                return !std::isspace(ch);
            }).base(),
            s.end());
}
static inline void trim(std::string &s)
{
    ltrim(s);
    rtrim(s);
}

void print_table(const std::vector<std::map<std::string, std::string>> &table)
{
    size_t i = 0;
    for (const auto &row : table)
    {
        std::cout << "----- " << std::setw(3) << i++ << std::endl;
        for (const auto &[name, value] : row)
        {
            std::cout << std::setw(12) << name << " : " << value << std::endl;
        }
    }
}

int input_number(const char *prompt)
{
    return std::stoi(input_string(prompt));
}
unsigned int input_unsigned(const char *prompt)
{
    return std::stoul(input_string(prompt));
}
std::string input_string(const char *prompt)
{
    if (prompt != nullptr && prompt != '\0')
    {
        std::cout << prompt << std::endl;
    }
    std::cout << "> " << std::flush;
    std::string buf;
    std::getline(std::cin, buf);
    trim(buf);
    return buf;
}

std::string select_command(std::initializer_list<std::pair<const char *, const char *>> commands)
{
    while (true)
    {
        std::cout << "Choose one of the following:" << std::endl;
        for (const auto &[command, desc] : commands)
        {
            std::cout << command << " - " << desc << std::endl;
        }
        auto input = input_string(nullptr);
        for (const auto &[command, _] : commands)
        {
            if (input == command)
            {
                return input;
            }
        }
        std::cout << "Invalid input, try again." << std::endl;
    }
}
