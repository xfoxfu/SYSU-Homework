#include "ui.hpp"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <locale>
#include <fmt/core.h>
#include <fmt/color.h>

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
        fmt::print("----- {:>3} -----\n", i++);
        for (const auto &[name, value] : row)
        {
            fmt::print("{:>12} : {}\n", name, value);
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
    if (prompt != nullptr && *prompt != '\0')
        fmt::print(fmt::emphasis::bold | fg(fmt::color::dark_orange), "{} > ", prompt);
    else
        fmt::print(fmt::emphasis::bold | fg(fmt::color::dark_orange), "> ");
    std::string buf;
    std::getline(std::cin, buf);
    trim(buf);
    return buf;
}

std::string select_command(std::initializer_list<std::pair<const char *, const char *>> commands)
{
    while (true)
    {
        fmt::print("Choose one of the following:\n");
        for (const auto &[command, desc] : commands)
        {
            fmt::print("{} - {}\n", command, desc);
        }
        auto input = input_string(nullptr);
        for (const auto &[command, _] : commands)
        {
            if (input == command)
            {
                return input;
            }
        }
        fmt::print("Invalid input, try again.\n");
    }
}
