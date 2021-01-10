#ifndef VIEW_H
#define VIEW_H

#include "mysqlclient.h"

class View
{
public:
    View() {}
    void show(MySQLClient &client);

private:
    void choice();
    void stock(MySQLClient &client);
    void purchase(MySQLClient &client);
    void refund(MySQLClient &client);
    void report(MySQLClient &client);
};
#endif