#include "ui.hpp"

#include <algorithm>
#include <cctype>
#include <fmt/color.h>
#include <fmt/core.h>
#include <iomanip>
#include <iostream>
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
        fmt::print("----- {:>3} -----\n", i++);
        for (const auto &[name, value] : row)
        {
            fmt::print("{:>12} : {}\n", name, value);
        }
    }
}

int input_number(const char *prompt)
{
    while (true)
    {
        try
        {
            return std::stoi(input_string(prompt));
        }
        catch (const std::exception &e)
        {
            fmt::print(fg(fmt::color::red), "Invalid input.\n");
        }
    }
}
unsigned int input_unsigned(const char *prompt)
{
    while (true)
    {
        try
        {
            return std::stoul(input_string(prompt));
        }
        catch (const std::exception &e)
        {
            fmt::print(fg(fmt::color::red), "Invalid input.\n");
        }
    }
}
double input_double(const char *prompt)
{
    while (true)
    {
        try
        {
            return std::stod(input_string(prompt));
        }
        catch (const std::exception &e)
        {
            fmt::print(fg(fmt::color::red), "Invalid input.\n");
        }
    }
}
std::string input_string(const char *prompt)
{
    if (prompt != nullptr && *prompt != '\0')
        fmt::print(fmt::emphasis::bold | fg(fmt::color::dark_orange), "{}", prompt);
    fmt::print(fmt::emphasis::bold | fg(fmt::color::dark_orange), "> ");
    std::string buf;
    std::getline(std::cin, buf);
    if (buf.empty() && std::cin.eof())
    {
        fmt::print(fg(fmt::color::red), "Input reached EOF.\n");
        exit(1);
    }
    trim(buf);
    return buf;
}

/**
 * @brief  从给定的命令列表中令用户选择一个
 * @note   会持续运行直到用户输入有效的命令
 * @param  commands: 命令和解释的对
 * @retval 命令
 */
std::string select_command(std::initializer_list<std::pair<const char *, const char *>> commands)
{
    while (true)
    {
        fmt::print(fg(fmt::color::blue), "Choose one of the following:\n");
        for (const auto &[command, desc] : commands)
        {
            fmt::print(fmt::emphasis::underline | fg(fmt::color::dark_orange), "{}", command);
            fmt::print(" - {}\n", desc);
        }
        auto input = input_string(nullptr);
        for (const auto &[command, _] : commands)
        {
            if (input == command)
            {
                return input;
            }
        }
        fmt::print(fg(fmt::color::red), "Invalid input, try again.\n");
    }
}

void xlog::info(const char *message)
{
    fmt::print(fg(fmt::color::blue), message);
}
void xlog::success(const char *message)
{
    fmt::print(fg(fmt::color::green), message);
}
void xlog::fail(const char *message)
{
    fmt::print(fg(fmt::color::red), message);
}
