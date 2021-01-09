#include "refund.h"
#include <iostream>

int refund(MySQLClient &client, int order_id, int count)
{
    try
    {
        std::string sql = "select count from purchase where order_id = " + std::to_string(order_id);
        //std::cout << sql << std::endl;
        QueryResult result = client.query(sql.c_str());
        if (result.empty() || atoi((result[0]).at("count").c_str()) < count)
        {
            return -1;
        }
        sql = "insert into refund set order_id = " + std::to_string(order_id) + ", count = " + std::to_string(count);
        //std::cout << sql << std::endl;
        client.update(sql.c_str());
        sql = "update book set count = count + " + std::to_string(count) + " where book_id = (select book_id from purchase where order_id = " + std::to_string(order_id) + ")";
        //std::cout << sql << std::endl;
        client.update(sql.c_str());
        sql = "select max(refund_id) as max_id from refund where order_id = " + std::to_string(order_id) + " and count = " + std::to_string(count);
        //std::cout << sql << std::endl;

        result = client.query(sql.c_str());
        if (result.size())
        {
            return atoi((result[0]).at("max_id").c_str());
        }
        else
        {
            return -1;
        }
    }
    catch (MySQLException &e)
    {
        std::cout << "error: " << e.what() << std::endl;
        return -1;
    }
}

void testRefund(MySQLClient &client)
{
    //std::cout << refund(client, 1, 10) << std::endl;
    //std::cout << refund(client, 2, 1) << std::endl;
    //std::cout << refund(client, 4, 10) << std::endl;
    //std::cout << refund(client, 999, 1) << std::endl;
}