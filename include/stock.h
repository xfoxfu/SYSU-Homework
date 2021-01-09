#ifndef STOCK_H
#define STOCK_H

#include "mysqlclient.h"
/**
 * 显示某本书的库存情况
 * @param title 图书名称
 * @return 图书名称对应的库存信息
 */
std::vector<std::map<std::string, std::string>> showCurrentCount(MySQLClient &client, std::string title);

/**
 * 显示书的供应商情况
 * @param title 图书名称
 * @return 图书名称对应的供货信息
 */
std::vector<std::map<std::string, std::string>> showProviderForBook(MySQLClient &client, std::string title);

/**
 * 进货并更新库存
 * @param offer_id: 供应单id
 * @param book_id: 图书id
 * @param amount: 进货数量
 */
void increaseStock(MySQLClient &client, int offer_id, int book_id, int amount);

// 示例前端
void fakeFrontEnd(MySQLClient &client);
#endif