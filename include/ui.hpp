#pragma once

#include <map>
#include <string>
#include <vector>
#include <utility>

void print_table(const std::vector<std::map<std::string, std::string>> &table);

int input_number(const char *prompt);
unsigned int input_unsigned(const char *prompt);
std::string input_string(const char *prompt);

std::string select_command(std::initializer_list<std::pair<const char *, const char *>> commands);
