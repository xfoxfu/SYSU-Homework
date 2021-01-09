#pragma once
#include "mysqlclient.h"
bool purchase(MySQLClient &c, int book_id, std::string name,int purchase_num);

bool getReportByMonth(MySQLClient& c, int month);

bool getRankByYear(MySQLClient& c);
