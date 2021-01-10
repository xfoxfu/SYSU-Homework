#include "refund.h"
#include <iostream>

QueryResult refund(MySQLClient &client, int order_id, int count)
{
    try
    {
        std::string sql = "select count from purchase where order_id = " + std::to_string(order_id);
        //std::cout << sql << std::endl;
        QueryResult result = client.query(sql.c_str());
        if (result.empty() || atoi((result[0]).at("count").c_str()) < count)
        {
            return QueryResult();
        }
        sql = "insert into refund set order_id = " + std::to_string(order_id) + ", count = " + std::to_string(count);
        //std::cout << sql << std::endl;
        client.update(sql.c_str());
        sql = "update book set count = count + " + std::to_string(count) + " where book_id = (select book_id from purchase where order_id = " + std::to_string(order_id) + ")";
        //std::cout << sql << std::endl;
        client.update(sql.c_str());
        sql = "select * from refund where order_id = " + std::to_string(order_id) + " and count = " + std::to_string(count) + " order by refund_id";
        //std::cout << sql << std::endl;

        result = client.query(sql.c_str());
        if (result.empty())
        {
            return QueryResult();
        }
        else
        {
            QueryResult ans;
            ans.push_back(result[0]);
            return ans;
        }
    }
    catch (MySQLException &e)
    {
        std::cout << "error: " << e.what() << std::endl;
        return QueryResult();
    }
}

void testRefund(MySQLClient &client)
{
    //std::cout << refund(client, 1, 10) << std::endl;
    //std::cout << refund(client, 2, 1) << std::endl;
    //std::cout << refund(client, 4, 10) << std::endl;
    //std::cout << refund(client, 999, 1) << std::endl;
}