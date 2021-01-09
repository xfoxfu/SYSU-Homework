#include "stock.h"
#include <string>
#include <exception>

std::vector<std::map<std::string, std::string>> showCurrentCount(MySQLClient &client, std::string title)
{
    try
    {
        std::string sql = "select * from book where title = '" + title + "'";
        return client.query(sql.c_str());
    }
    catch (MySQLException &e)
    {
        std::cout << "error: " << e.what() << std::endl;
        return std::vector<std::map<std::string, std::string>>();
    }
}

std::vector<std::map<std::string, std::string>> showProviderForBook(MySQLClient &client, std::string title)
{
    try
    {
        std::string sql = "select provider_id, book.book_id, title, offer.price from book, offer where title = '" + title + "' and book.book_id = offer.book_id";
        std::cout << sql << std::endl;
        return client.query(sql.c_str());
    }
    catch (MySQLException &e)
    {
        std::cout << "error: " << e.what() << std::endl;
        return std::vector<std::map<std::string, std::string>>();
    }
}

void increaseStock(MySQLClient &client, int offer_id, int book_id, int amount)
{
    try
    {
        std::string insert, update;
        insert = "insert into stock set offer_id = " + std::to_string(offer_id) + ", count = " + std::to_string(amount);
        update = "update book set count = count + " + std::to_string(amount) + " where book_id = " + std::to_string(book_id);
        client.update(insert.c_str());
        client.update(update.c_str());
    }
    catch (MySQLException &e)
    {
        std::cout << "error: " << e.what() << std::endl;
        return;
    }
}

#include <iostream>

void fakeFrontEnd(MySQLClient &client)
{
    /**
     * 调用showCurrentCount显示库存情况
     * 调用showProviderForBook显示供应商情况
     * 调用stock进货
     * std::vector<std::map<std::string, std::string>>是表的数据，用iterator遍历即可
     */
}