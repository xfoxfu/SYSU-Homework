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
    const char *host = "127.0.0.1";
    const char *port = "50500";
    cout << "Usage: " << argv[0] << " [server=" << host << "] [port=" << port << "]" << endl;
    if (argc >= 2)
        host = argv[1];
    if (argc >= 3)
        port = argv[2];
    cout << "Connecting to " << host << ":" << port << endl;
    fox_socket sock(host, port);
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

            if (msg == "exit")
            {
                cout << "Leave!" << endl;
                sock.close();
                th.join();
                break;
            }
        }
    }
    catch (std::exception &e)
    {
        cout << e.what() << endl;
    }
}

void handle(fox_socket sock)
{
    try
    {
        while (true)
        {
            auto str = sock.recv();
            cout << '\r' << str << endl
                 << PROMPT << std::flush;
        }
    }
    catch (const std::exception &ex)
    {
        if (!(strstr(ex.what(), "Bad file descriptor") != nullptr))
        {
            throw ex;
        }
    }
}
