#include <iostream>
#include "mysqlclient.h"
#include "stock.h"

int main(int argc, char **argv)
{
    // 后面直接使用client::update, query方法即可执行增删查改
    std::string host, port, user, password, database;
    if (argc != 6)
        return 0;

    host = argv[1], port = argv[2], user = argv[3], password = argv[4], database = argv[5];
    MySQLClient client(host.c_str(), atoi(port.c_str()), user.c_str(), password.c_str(), database.c_str());

    /*
    std::string command = select_command(
        {
            {"test1", "This is description"},
            {"test2", "This is description 2"},
        });
        */
    // 入库示例
    fakeFrontEnd(client);
    return 0;
}
