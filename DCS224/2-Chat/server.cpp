#include <thread>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <queue>
#include <string>
#include <list>
#include <utility>
#include <chrono>
#include "socket.hpp"

using std::cout;
using std::endl;
using std::list;
using std::mutex;
using std::pair;
using std::queue;
using std::string;
using std::thread;
using std::chrono::system_clock;

list<pair<fox_socket, thread>> sockets;
mutex sockets_lock;

mutex stdio_lock;

void handler(fox_socket sock);
void broadcast(const string &content);

int main()
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

            {
                auto _guard = std::lock_guard(sockets_lock);
                sockets.push_back(make_pair(std::move(sub_sock), thread(handler, sub_sock)));
            }
        }
    }
    catch (std::exception &e)
    {
        cout << e.what() << endl;
    }
    sock.close();
    return 0;
}

void handler(fox_socket sock)
{
    while (true)
    {
        auto str = sock.recv();
        broadcast(str);
    }
}

void broadcast(const string &content)
{
    {
        auto _guard_stdio = std::lock_guard(stdio_lock);
        cout << content << endl;
    }
    {
        auto _guard = std::lock_guard(sockets_lock);
        for (const auto &[sock, _] : sockets)
        {
            sock.send(content);
        }
    }
}
