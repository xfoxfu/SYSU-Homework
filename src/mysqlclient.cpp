#include "mysqlclient.h"
#include "../vendor/bprinter/table_printer.h"
#include "fmt/color.h"
#include "fmt/core.h"
#include "libmysql/mysqld_error.h"
#include <vector>

std::map<std::string, int> MySQLClient::col_size = std::map<std::string, int>{
    {"book_id", 8},
    {"title", 30},
    {"author", 15},
    {"isbn", 10},
    {"count", 5},
    {"price", 10},
    {"created_at", 20},
    {"updated_at", 20},
    {"total_cost", 12},
    {"month", 8},
    {"provider_id", 8},

    {"refund_id", 8},
    {"order_id", 8},
    {"stock_id", 8},
    {"name", 15},
    {"phone", 15},
    {"offer_id", 4},
    {"offer_price", 10},
};

MySQLClient::MySQLClient(const char *host, unsigned int port,
                         const char *user, const char *password,
                         const char *database)
{
    mysql_init(&mysql);
    this->host = host;
    this->port = port;
    this->user = user;
    this->password = password;
    this->database = database;
    openConnection();
}

MySQLClient::~MySQLClient()
{
    releaseConnection();
}

long long MySQLClient::update(const char *sql)
{
    int queryResult = mysql_query(&mysql, sql);
    if (queryResult)
    {
        throw MySQLException(mysql_errno(&mysql), mysql_error(&mysql));
    }

    return mysql_affected_rows(&mysql);
}
std::vector<std::map<std::string, std::string>> MySQLClient::query(const char *sql)
{
    std::vector<std::map<std::string, std::string>> ans;
    int queryResult = mysql_query(&mysql, sql);
    if (queryResult)
    {
        throw MySQLException(mysql_errno(&mysql), mysql_error(&mysql));
    }

    MYSQL_RES *result = mysql_store_result(&mysql);
    unsigned int size = mysql_field_count(&mysql);
    MYSQL_FIELD *fields = mysql_fetch_fields(result);
    MYSQL_ROW row;
    while ((row = mysql_fetch_row(result)) != nullptr)
    {
        std::map<std::string, std::string> temp;
        for (int i = 0; i < size; ++i)
        {
            temp[fields[i].name] = row[i] ? row[i] : "null";
        }
        ans.push_back(temp);
    }
    while (mysql_next_result(&mysql) != -1)
        ;
    mysql_free_result(result);
    return ans;
}

void MySQLClient::printTable(std::vector<std::string> keys, std::vector<std::map<std::string, std::string>> &t)
{
    bprinter::TablePrinter bt(&std::cout);
    bt.set_flush_left();
    for (const auto &key : keys)
    {
        bt.AddColumn(key, col_size[key]);
    }
    bt.PrintHeader();
    for (auto row : t)
    {
        for (const auto &key : keys)
        {
            bt << row[key];
        }
    }
    bt.PrintFooter();
}

void MySQLClient::openConnection()
{
    MYSQL *ans = mysql_real_connect(&mysql, host.c_str(), user.c_str(), password.c_str(), database.c_str(), port, nullptr, CLIENT_MULTI_STATEMENTS | CLIENT_MULTI_QUERIES | CLIENT_MULTI_RESULTS);
    if (!ans)
    {
        throw MySQLException(mysql_errno(&mysql), mysql_error(&mysql));
    }
}

void MySQLClient::releaseConnection()
{
    mysql_close(&mysql);
}

MySQLException::MySQLException(unsigned int code, const char *error)
{
    this->code = code;
    this->error = error;
}
const std::string &MySQLException::what()
{
    return error;
}
const char *MySQLException::what() const noexcept
{
    return error.c_str();
}
unsigned int MySQLException::ecode() const noexcept
{
    return code;
}

bool MySQLException::is_fk_no_ref() const noexcept
{
    return code == ER_NO_REFERENCED_ROW || code == ER_NO_REFERENCED_ROW_2;
}

bool MySQLException::is_check_fail() const noexcept
{
    return code == ER_CHECK_CONSTRAINT_VIOLATED;
}

void MySQLException::print() const noexcept
{
    fmt::print(fg(fmt::color::red), "MySQL Exception: ({}) {}\n", ecode(), what());
}
