#include "mysqlclient.h"
#include "table.h"
#include <algorithm>
#include <initializer_list>
#include <string>
using namespace std;
struct RetrieveKey
{
    template <typename T>
    typename T::first_type operator()(T keyValuePair) const
    {
        return keyValuePair.first;
    }
};
struct RetrieveVal
{
    template <typename T>
    typename T::second_type operator()(T keyValuePair) const
    {
        return keyValuePair.second;
    }
};

bool purchase(MySQLClient c, int book_id, int purchase_num)
{
    string query_str = "set @res=0; call purchase('" + to_string(book_id) + "'," + to_string(purchase_num) + ",@res);";
    auto retTable = c.query(query_str.c_str())[0];
    query_str = "select @res;";
    auto resStr = c.query(query_str.c_str());
    bool res = stoi(resStr[0].begin()->second);
    if (!res)
        return res;
    vector<string> keys;
    initializer_list<string> values;
    std::transform(retTable.begin(), retTable.end(), back_inserter(keys), RetrieveKey());
    std::transform(retTable.begin(), retTable.end(), back_inserter(values), RetrieveVal());
    VariadicTable<string, string, string, string, string, string, string, string, string> vt(keys);

    vt.addRow(retTable[keys[1]], retTable[keys[2]], retTable[keys[3]], retTable[keys[4]], retTable[keys[5]], retTable[keys[6]], retTable[keys[7]], retTable[keys[8]], retTable[keys[9]]);
    vt.print(std::cout);
    return res;
}
