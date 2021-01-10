#include "controller/offer.h"
#include "controller/book.h"
#include "controller/provider.h"
#include "ui.hpp"
#include <fmt/core.h>

void offer::menu(MySQLClient &client)
{
    auto op = select_command({
        {"list", "List all offer entries."},
        {"query", "Find offer entries by book name, isbn, provider name or phone."},
        {"create", "Create a new offer entry."},
        {"update", "Update a present offer entry."},
        {"delete", "Delete a offer entry."},
    });
    if (op == "list")
        offer::list(client);
    if (op == "query")
        offer::query(client);
    else if (op == "create")
        offer::create(client);
    else if (op == "update")
        offer::update(client);
    else if (op == "delete")
        offer::remove(client);
}

void offer::list(MySQLClient &client)
{
    auto sql = fmt::format(
        "SELECT book.*, provider.*, offer.offer_id, offer.price AS offer_price FROM offer "
        "LEFT JOIN book ON book.book_id = offer.book_id "
        "LEFT JOIN provider ON provider.provider_id = offer.provider_id; ");
    auto res = client.query(sql.c_str());
    client.printTable(
        {"offer_id", "book_id", "title", "author", "isbn", "provider_id", "name", "phone", "price", "offer_price"},
        res);
}

void offer::query(MySQLClient &client)
{
    auto query = input_string("Search");
    auto sql = fmt::format(
        "SELECT book.*, provider.*, offer.offer_id, offer.price AS offer_price FROM offer "
        "LEFT JOIN book ON book.book_id = offer.book_id "
        "LEFT JOIN provider ON provider.provider_id = offer.provider_id "
        "WHERE book.title LIKE '%{0}%' OR book.isbn LIKE '%{0}%' "
        "OR provider.name LIKE '%{0}%' OR provider.phone LIKE '%{0}%'; ",
        query);
    auto res = client.query(sql.c_str());
    client.printTable(
        {"offer_id", "book_id", "title", "author", "isbn", "provider_id", "name", "phone", "price", "offer_price"},
        res);
}

void offer::create(MySQLClient &client)
{
    xlog::info("Select a book from the following list by search\n");
    book::query(client);
    auto book_id = input_number("book_id");

    xlog::info("Select a provider from the following list by search\n");
    provider::query(client);
    auto provider_id = input_number("provider_id");

    auto price = input_double("Price");

    auto sql = fmt::format(
        "INSERT INTO offer (book_id, provider_id, price) "
        "VALUES ({}, {}, {})",
        book_id, provider_id, price);
    try
    {
        auto res = client.update(sql.c_str());
    }
    catch (const MySQLException &ex)
    {
        ex.print();
        if (ex.is_fk_no_ref())
        {
            xlog::fail("`book_id` or `provider_id` is not valid.\n");
            return;
        }
        else
        {
            throw ex;
        }
    }

    xlog::success("Offer created successfully.\n");
}

void offer::update(MySQLClient &client)
{
    xlog::info("Select a offer from the following list by search\n");
    offer::query(client);
    auto id = input_number("offer_id");

    auto value = input_double("New Price");
    auto sql = fmt::format("UPDATE {} SET {} = {} WHERE offer_id = {};", "offer", "price", value, id);
    auto res = client.update(sql.c_str());

    xlog::success("Offer successfully updated.\n");
}

void offer::remove(MySQLClient &client)
{
    xlog::info("Select a offer from the following list by search\n");
    offer::query(client);
    auto id = input_number("offer_id");

    auto sql = fmt::format("DELETE FROM {} WHERE offer_id = {};", "offer", id);
    auto res = client.update(sql.c_str());

    xlog::success("Offer successfully removed.\n");
}
