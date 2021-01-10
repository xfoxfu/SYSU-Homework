#include "controller/book.h"
#include "ui.hpp"
#include <fmt/core.h>

void book::menu(MySQLClient &client)
{
    auto op = select_command({
        {"list", "List all book entries."},
        {"query", "Find book entries by name or ISBN."},
        {"create", "Create a new book entry."},
        {"update", "Update a present book entry."},
        {"delete", "Delete a book entry."},
    });
    if (op == "list")
        book::list(client);
    if (op == "query")
        book::query(client);
    else if (op == "create")
        book::create(client);
    else if (op == "update")
        book::update(client);
    else if (op == "delete")
        book::remove(client);
}

void book::list(MySQLClient &client)
{
    auto sql = fmt::format("SELECT * FROM {};", "book");
    auto res = client.query(sql.c_str());
    client.printTable(
        {"book_id", "title", "author", "isbn", "count", "price", "created_at", "updated_at"},
        res);
}

void book::query(MySQLClient &client)
{
    auto query = input_string("Search");
    auto sql = fmt::format("SELECT * FROM {} "
                           "WHERE title LIKE '%{}%' OR isbn LIKE '%{}%';",
                           "book", query, query);
    auto res = client.query(sql.c_str());
    client.printTable(
        {"book_id", "title", "author", "isbn", "count", "price", "created_at", "updated_at"},
        res);
}

void book::create(MySQLClient &client)
{
    auto title = input_string("Title");
    auto author = input_string("Author");
    auto isbn = input_string("ISBN");
    auto count = 0;
    auto price = input_string("Price");

    auto sql = fmt::format(
        "INSERT INTO book (title, author, isbn, count, price) "
        "VALUES (\"{}\", \"{}\", \"{}\", {}, {})",
        title, author, isbn, count, price);
    auto res = client.query(sql.c_str());

    log::success("Book created successfully.");
}

void book::update(MySQLClient &client)
{
    log::info("Select a book from the following list");
    book::list(client);
    auto id = input_number("book_id");

    auto field = select_command({
        {"title", "Title"},
        {"author", "Author"},
        {"isbn", "ISBN"},
        {"price", "Price"},
    });
    std::string sql;
    if (field != "price")
    {
        auto value = input_string(field.c_str());
        sql = fmt::format("UPDATE {} SET {} = \"{}\" WHERE book_id = {};", "book", field, value, id);
    }
    else
    {
        auto value = input_string(field.c_str());
        sql = fmt::format("UPDATE {} SET {} = {} WHERE book_id = {};", "book", field, value, id);
    }
    auto res = client.query(sql.c_str());

    log::success("Book successfully updated.");
}

void book::remove(MySQLClient &client)
{
    log::info("Select a book from the following list");
    book::list(client);
    auto id = input_number("book_id");

    auto sql = fmt::format("DELETE FROM {} WHERE book_id = {};", "book", id);
    auto res = client.query(sql.c_str());

    log::success("Book successfully removed.");
}
