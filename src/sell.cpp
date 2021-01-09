#include "sell.h"
#include <algorithm>
#include <initializer_list>
#include <string>
using namespace std;

/// <summary>
/// 买书
/// </summary>
/// <param name="c">mysql对象</param>
/// <param name="book_id">书id</param>
/// <param name="name">购书者名字</param>
/// <param name="purchase_num">购买量</param>
/// <returns></returns>
bool purchase(MySQLClient &c, int book_id,string name, int purchase_num)
{
    try
    {
        string query_str = "select title,count from book where book_id="+to_string(book_id);
        auto retTable = c.query(query_str.c_str());
        c.printTable({ "title","count" }, retTable);

        query_str = "call purchase(" + to_string(book_id) + "," + to_string(purchase_num) +", '"+name+ "')";
        retTable = c.query(query_str.c_str());
        if (retTable.size() == 0)
            return false;
        c.printTable({ "book_id","title","author","isbn","count","price","created_at","updated_at","total_cost" },retTable);
    }
    catch (const std::exception&e)
    {
        cout << e.what()<<endl;
        return false;
    }
    return true;
}

/// <summary>
/// 每月的排行榜，按销售本数排序
/// </summary>
/// <param name="c">mysql对象</param>
/// <param name="month">月份</param>
bool getReportByMonth(MySQLClient& c, int month){
    string queryStr ="select book.book_id, book.title,sum(purchase.count) as count,sum(purchase.price) as price from purchase join book on purchase.book_id=book.book_id where month(purchase.created_at)= "+to_string(month)+ " group by purchase.book_id order by count desc;";
    try
    {
        auto retTable=c.query(queryStr.c_str());
        c.printTable({"book_id","title","count","price"}, retTable);
    }
    catch (const std::exception&e)
    {
        cout << e.what() << endl;
        return false;
    }
    return true;
}

/// <summary>
/// 回溯整年的销售总额和总量
/// </summary>
/// <param name="c"></param>
/// <returns></returns>
bool getRankByYear(MySQLClient& c){
    try
    {
        string queryStr ="select date_format(M.month, '%Y-%m') as month, COALESCE(sum(price), 0) as price,COALESCE(sum(count), 0) as count from purchase right outer join (SELECT	DATE_FORMAT( @cdate := DATE_ADD( @cdate, INTERVAL + 1 MONTH ), '%y-%m-%d' ) AS month FROM	( SELECT @cdate := DATE_ADD( CURRENT_DATE, INTERVAL - 1 YEAR ) FROM purchase LIMIT 12 ) t0 WHERE	date( @cdate ) <= DATE_ADD( CURRENT_DATE, INTERVAL - 1 DAY ))as M on month(purchase.updated_at)=month(M.month) group by M.month order by M.month;";
        auto retTable=c.query(queryStr.c_str());
        c.printTable({"month","price","count"}, retTable);
    }
    catch (const std::exception&e)
    {
        cout << e.what() << endl;
        return false;
    }
    return true;
}
