#include "sell.h"
#include <algorithm>
#include <initializer_list>
#include <string>
using namespace std;


bool purchase(MySQLClient &c, int book_id,string name, int purchase_num)
{
    string query_str = "select title,count from book where book_id="+to_string(book_id);
    auto retTable = c.query(query_str.c_str());
    c.printTable({ "title","count" }, retTable);

    query_str = "call purchase(" + to_string(book_id) + "," + to_string(purchase_num) +", '"+name+ "')";
    retTable = c.query(query_str.c_str());
    if (retTable.size() == 0)
        return false;
    c.printTable({ "book_id","title","author","isbn","count","price","created_at","updated_at","total_cost" },retTable);
    return true;
}
