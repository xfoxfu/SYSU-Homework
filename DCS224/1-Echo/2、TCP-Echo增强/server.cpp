#include <chrono>
#include <sstream>
#include <iostream>
#include <iomanip>
#include "socket.hpp"

using std::cout;
using std::endl;
using std::chrono::system_clock;

int main(int argc, char *argv[])
{
    fox_socket sock(INADDR_ANY, 3899);
    try
    {
        sock.bind();
        cout << "Server started." << endl
             << endl;

        while (1)
        {
            auto sub_sock = sock.accept();
            // cout << "Accepted." << endl;

            auto msg = sub_sock.recv();
            cout << "Received: " << msg << endl;

            auto time = system_clock::to_time_t(system_clock::now());
            cout << "At: " << std::put_time(std::localtime(&time), "%F %T") << endl;

            int ip_segments[4] = {0};
            ip_segments[0] = sub_sock.ip() << 24 >> 24;
            ip_segments[1] = sub_sock.ip() << 16 >> 24;
            ip_segments[2] = sub_sock.ip() << 8 >> 24;
            ip_segments[3] = sub_sock.ip() >> 24;

            cout << "Client IP: " << ip_segments[0] << "." << ip_segments[1] << "." << ip_segments[2] << "." << ip_segments[3] << endl
                 << "Client Port: " << sub_sock.port() << endl
                 << endl;

            std::ostringstream os;
            os << "Content: " << msg << endl
               << "At: " << std::put_time(std::localtime(&time), "%F %T") << endl
               << "Client IP: " << ip_segments[0] << "." << ip_segments[1] << "." << ip_segments[2] << "." << ip_segments[3] << endl
               << "Client Port: " << sub_sock.port() << endl;
            sub_sock.send(os.str());

            sub_sock.close();
        }
    }
    catch (std::exception &e)
    {
        cout << e.what() << endl;
    }
    sock.close();
}