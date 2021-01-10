#include "view.h"
#include "stock.h"
#include "sell.h"
#include "refund.h"

#include <iostream>
#include <string>

void View::choice()
{
    std::cout << "---please enter your choice---\n"
              << "1 stock\n"
              << "2 refund\n"
              << "3 report\n"
              << "4 purchase\n"
              << "5 bookstore management\n"
              << "6 offer management\n"
              << "7 provider management\n"
              << "0 exit\n"
              << "------------------------------\n"
              << "? ";
}

void View::show(MySQLClient &client)
{
    std::string input;
    int item;

    while (true)
    {
        choice();
        std::getline(std::cin, input);
        item = atoi(input.c_str());
        switch (item)
        {
        case 1:
            stock(client);
            break;
        case 2:
            refund(client);
            break;
        case 3:
            report(client);
            break;
        case 4:
            purchase(client);
            break;
        case 0:
            std::cout << "\nthank you, bye\n";
            exit(0);
            break;
        default:
            std::cout << "invalid input!\n";
            break;
        }
        std::cout << "press enter to continue.......\n";
        std::getline(std::cin, input);
    }
}

void View::stock(MySQLClient &client)
{
    std::string input;
    std::cout << "enter the book title: ";
    std::getline(std::cin, input);
    QueryResult result = showCurrentCount(client, input);
    if (result.empty())
    {
        std::cout << "invalid input!\n";
        return;
    }

    std::cout << "book '" << input << "' in the bookstore\n";
    client.printTable({"book_id", "title", "author", "isbn", "count", "price", "created_at", "updated_at"}, result);

    std::cout << "offers for the book '" << input << "'\n";
    result = showProviderForBook(client, input);
    client.printTable({"provider_id", "book_id", "title", "price"}, result);

    std::cout << "input the provider_id, 0 for exit: ";
    std::getline(std::cin, input);
    if (input == "0")
        return;
    int offer_id = atoi(input.c_str());

    std::cout << "input the book_id, 0 for exit: ";
    std::getline(std::cin, input);
    if (input == "0")
        return;
    int book_id = atoi(input.c_str());

    std::cout << "input the amount, 0 for exit: ";
    std::getline(std::cin, input);
    if (input == "0")
        return;
    int amount = atoi(input.c_str());
    result = increaseStock(client, offer_id, book_id, amount);
    client.printTable({"stock_id", "offer_id", "count", "created_at", "updated_at"}, result);
}

void View::purchase(MySQLClient &client)
{
    std::string input;

    std::cout << "input the book_id, 0 for exit: ";
    std::getline(std::cin, input);
    if (input == "0")
        return;
    int book_id = atoi(input.c_str());

    std::cout << "input the customer name, 0 for exit: ";
    std::getline(std::cin, input);
    if (input == "0")
        return;
    std::string name = input;

    std::cout << "input the amount, 0 for exit: ";
    std::getline(std::cin, input);
    if (input == "0")
        return;
    int amount = atoi(input.c_str());

    ::purchase(client, book_id, name, amount);
}
void View::refund(MySQLClient &client)
{
    std::string input;

    std::cout << "input the offer_id, 0 for exit: ";
    std::getline(std::cin, input);
    if (input == "0")
        return;
    int offer_id = atoi(input.c_str());

    std::cout << "input the amount, 0 for exit: ";
    std::getline(std::cin, input);
    if (input == "0")
        return;
    int amount = atoi(input.c_str());

    QueryResult result = ::refund(client, offer_id, amount);
    client.printTable({"refund_id", "order_id", "count", "created_at", "updated_at"}, result);
}
void View::report(MySQLClient &client)
{
    std::string input;
    std::cout << "---please enter report type---\n"
              << "1 month report\n"
              << "2 year report\n"
              << "0 exit\n"
              << "------------------------------\n"
              << "? ";
    std::getline(std::cin, input);
    int item = atoi(input.c_str());
    switch (item)
    {
    case 0:
        return;
        break;

    case 1:
        std::cout << "enter the month: ";
        std::getline(std::cin, input);
        item = atoi(input.c_str());
        getReportByMonth(client, item);
        break;

    case 2:
        getRankByYear(client);
        break;

    default:
        std::cout << "invalid input\n";
        break;
    }
}