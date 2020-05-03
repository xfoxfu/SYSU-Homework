#include <chrono>
#include <sstream>
#include <iostream>
#include <iomanip>
#include "socket.hpp"

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::chrono::system_clock;

int main(int argc, char *argv[])
{
    fox_socket sock(inet_addr("127.0.0.1"), 3899);
    try
    {
        sock.connect();
        cout << "Connected." << endl;

        string msg;
        cout << "Input string to send: ";
        cin >> msg;

        sock.send(msg);

        auto recv = sock.recv();
        cout << "Received: " << endl
             << recv << endl;
    }
    catch (std::exception &e)
    {
        cout << e.what() << endl;
    }
    sock.close();
}
