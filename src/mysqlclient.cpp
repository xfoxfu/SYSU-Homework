#include "mysqlclient.h"

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

MySQLClient::~MySQLClient() {
    releaseConnection();
}

long long MySQLClient::update(const char *sql)
{
    int queryResult = mysql_query(&mysql, sql);
    if (queryResult)
    {
        throw MySQLException(mysql_error(&mysql));
    }

    return mysql_affected_rows(&mysql);
}
std::vector<std::map<std::string, std::string>> MySQLClient::query(const char *sql)
{
    std::vector<std::map<std::string, std::string>> ans;
    int queryResult = mysql_query(&mysql, sql);
    if (queryResult)
    {
        throw MySQLException(mysql_error(&mysql));
    }

    MYSQL_RES *result = mysql_use_result(&mysql);
    unsigned int size = mysql_field_count(&mysql);
    MYSQL_FIELD *fields = mysql_fetch_fields(result);
    MYSQL_ROW row;
    while (row = mysql_fetch_row(result))
    {
        std::map<std::string, std::string> temp;
        for (int i = 0; i < size; ++i)
        {
            temp[fields[i].name] = row[i] ? row[i] : "null";
        }
        ans.push_back(temp);
    }

    mysql_free_result(result);

    return ans;
}

void MySQLClient::openConnection()
{
    MYSQL *ans = mysql_real_connect(&mysql, host.c_str(), user.c_str(), password.c_str(), database.c_str(), port, nullptr, 0);
    if (!ans)
    {
        throw MySQLException(mysql_error(&mysql));
    }
}

void MySQLClient::releaseConnection()
{
    mysql_close(&mysql);
}

MySQLException::MySQLException(const char *error)
{
    this->error = error;
}
const std::string &MySQLException::what()
{
    return error;
}