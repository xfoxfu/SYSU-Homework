#include <chrono>
#include <sstream>
#include <iostream>
#include <thread>
#include <iomanip>
#include "socket.hpp"

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::thread;
using std::chrono::system_clock;

const char *PROMPT = ">> ";

void handle(fox_socket sock);

int main(int argc, char *argv[])
{
    fox_socket sock(inet_addr("127.0.0.1"), 3899);
    cout << "103.26.79.35" << endl;
    try
    {
        sock.connect();
        cout << "Connected." << endl;

        auto th = thread(handle, sock);

        while (true)
        {
            string msg;
            cout << PROMPT << std::flush;
            cin >> msg;

            sock.send(msg);
        }
    }
    catch (std::exception &e)
    {
        cout << e.what() << endl;
    }
    sock.close();
}

void handle(fox_socket sock)
{
    while (true)
    {
        auto str = sock.recv();
        cout << '\r' << str << endl
             << PROMPT << std::flush;
    }
}
