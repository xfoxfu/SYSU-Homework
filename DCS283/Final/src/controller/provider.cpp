#include "controller/provider.h"
#include "ui.hpp"
#include <fmt/core.h>

void provider::menu(MySQLClient &client)
{
    auto op = select_command({
        {"list", "List all provider entries."},
        {"query", "Find provider entries by name or phone."},
        {"create", "Create a new provider entry."},
        {"update", "Update a present provider entry."},
        {"delete", "Delete a provider entry."},
    });
    if (op == "list")
        provider::list(client);
    if (op == "query")
        provider::query(client);
    else if (op == "create")
        provider::create(client);
    else if (op == "update")
        provider::update(client);
    else if (op == "delete")
        provider::remove(client);
}

void provider::list(MySQLClient &client)
{
    auto sql = fmt::format("SELECT * FROM {};", "provider");
    auto res = client.query(sql.c_str());
    client.printTable(
        {"provider_id", "name", "phone", "created_at", "updated_at"},
        res);
}

void provider::query(MySQLClient &client)
{
    auto query = input_string("Search");
    auto sql = fmt::format("SELECT * FROM {} "
                           "WHERE name LIKE '%{}%' OR phone LIKE '%{}%';",
                           "provider", query, query);
    auto res = client.query(sql.c_str());
    client.printTable(
        {"provider_id", "name", "phone", "created_at", "updated_at"},
        res);
}

void provider::create(MySQLClient &client)
{
    auto name = input_string("Name");
    auto phone = input_string("Phone");

    auto sql = fmt::format(
        "INSERT INTO provider (name, phone) "
        "VALUES (\"{}\", \"{}\")",
        name, phone);
    auto res = client.update(sql.c_str());

    xlog::success("Provider created successfully.\n");
}

void provider::update(MySQLClient &client)
{
    xlog::info("Select a provider from the following list\n");
    provider::list(client);
    auto id = input_number("provider_id");

    auto field = select_command({
        {"name", "Name"},
        {"phone", "Phone"},
    });
    auto value = input_string(field.c_str());
    auto sql = fmt::format("UPDATE {} SET {} = \"{}\" WHERE provider_id = {};", "provider", field, value, id);
    auto res = client.update(sql.c_str());

    xlog::success("Provider successfully updated.\n");
}

void provider::remove(MySQLClient &client)
{
    xlog::info("Select a provider from the following list\n");
    provider::list(client);
    auto id = input_number("provider_id");

    auto sql = fmt::format("DELETE FROM {} WHERE provider_id = {};", "provider", id);
    auto res = client.update(sql.c_str());

    xlog::success("Provider successfully removed.\n");
}
