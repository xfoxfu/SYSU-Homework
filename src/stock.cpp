#include "stock.h"
#include <string>

std::vector<std::map<std::string, std::string>> showCurrentCount(MySQLClient client, std::string title)
{
    std::string sql = "select * from book where title = '" + title + "'";
    return client.query(sql.c_str());
}

std::vector<std::map<std::string, std::string>> showProviderForBook(MySQLClient client, std::string title)
{
    std::string sql = "select * from book natural join offer where title = '" + title + "'";
    return client.query(sql.c_str());
}

void stock(MySQLClient client, int offer_id, int book_id, int amount)
{
    std::string insert, update;
    insert = "insert into stock set offer_id = " + to_string(offer_id) + ", count = " + to_string(amount);
    update = "update book set count = count + " + to_string(amount) + " where book_id = " + to_string(book_id);
    client.update(insert.c_str());
    client.update(update.c_str());
}
// 示例前端
void stockFakeFrontEnd()
{
    // 先调用showCurrentCount显示库存情况
}