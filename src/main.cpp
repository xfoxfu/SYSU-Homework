#include <iostream>
#include "ui.hpp"
#include <fmt/core.h>

int main(int argc, char **argv)
{
    std::string command = select_command(
        {
            {"test1", "This is description"},
            {"test2", "This is description 2"},
        });
    return 0;
}
