#include "controller/book.h"
#include "mysqlclient.h"
#include "refund.h"
#include "sell.h"
#include "stock.h"
#include <iostream>

int main(int argc, char **argv)
{
    // 后面直接使用client::update, query方法即可执行增删查改
    std::string host, port, user, password, database;
    if (argc != 6)
    {
        host = "localhost";
        port = "3306";
        user = "root";
        password = "0000";
        database = "bookstore";
    }
    else
    {
        host = argv[1], port = argv[2], user = argv[3], password = argv[4], database = argv[5];
    }

    MySQLClient client(host.c_str(), atoi(port.c_str()), user.c_str(), password.c_str(), database.c_str());
    book::list(client);
    /*
    for (int i = 1; i < 10; i++)
    {
        purchase(client, i, std::string("zhenggehan"), 1);
    }
*/
    /*
    std::string command = select_command(
        {
            {"test1", "This is description"},
            {"test2", "This is description 2"},
        });
        */

    // 入库示例
    //fakeFrontEnd(client);
    // 退货测例
    testRefund(client);
    return 0;
}
