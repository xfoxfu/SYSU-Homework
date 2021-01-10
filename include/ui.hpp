#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

void print_table(const std::vector<std::map<std::string, std::string>> &table);

int input_number(const char *prompt);
unsigned int input_unsigned(const char *prompt);
double input_double(const char *prompt);
std::string input_string(const char *prompt);

std::string select_command(std::initializer_list<std::pair<const char *, const char *>> commands);

namespace xlog
{
    void info(const char *message);
    void success(const char *message);
    void fail(const char *message);
}; // namespace xlog
