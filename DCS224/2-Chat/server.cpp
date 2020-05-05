#include <thread>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <queue>
#include <string>
#include <list>
#include <utility>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <vector>
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
string make_message(const fox_socket &sock, const string &content);
void handle_disconnect(const fox_socket &sock) noexcept;

int main(int argc, char *argv[])
{
    const char *port = "50500";
    cout << "Usage: " << argv[0] << " [port=" << port << "]" << endl;
    if (argc >= 2)
    {
        port = argv[1];
    }
    cout << "Binding on 0.0.0.0:" << port << endl;
    fox_socket sock("0.0.0.0", port);
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
            broadcast(make_message(sub_sock, "Enter!"));
        }
    }
    catch (std::exception &e)
    {
        cout << "Exception: " << e.what() << endl;
    }
    sock.close();
    return 0;
}

void handler(fox_socket sock)
{
    while (sock.connected())
    {
        try
        {
            auto msg = sock.recv();
            broadcast(make_message(sock, msg));
        }
        catch (const std::exception &ex)
        {
            cout << "Error encountered while recv: " << ex.what() << endl;
            if (strstr(ex.what(), "Remote disconnected") != nullptr)
            {
                handle_disconnect(sock);
                // connected = false;
            }
        }
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
            if (!sock.connected())
            {
                continue;
            }
            try
            {
                sock.send(content);
            }
            catch (const std::exception &ex)
            {
                cout << "Error encountered while send: " << ex.what() << endl;
            }
        }
    }
}

string make_message(const fox_socket &sock, const string &content)
{
    std::ostringstream os;

    int ip_segments[4] = {0};
    ip_segments[0] = sock.ip() << 24 >> 24;
    ip_segments[1] = sock.ip() << 16 >> 24;
    ip_segments[2] = sock.ip() << 8 >> 24;
    ip_segments[3] = sock.ip() >> 24;
    auto time = system_clock::to_time_t(system_clock::now());

    os << "Client IP: " << ip_segments[0] << "." << ip_segments[1] << "." << ip_segments[2] << "." << ip_segments[3] << endl
       << "Client Port: " << sock.port() << endl
       << "Time: " << std::put_time(std::localtime(&time), "%F %T") << endl
       << "Message: " << content;
    return os.str();
}

void handle_disconnect(const fox_socket &sock) noexcept
{
    {
        auto _guard = std::lock_guard(sockets_lock);
        for (auto it = sockets.begin(); it != sockets.end(); it++)
        {
            if (it->first == sock)
            {
                // Here the thread cannot be joined, because it may be just the current thread.
                // it->second.join();
                it->first.close();
                // Cannot remove the iterator, because it may refer to current thread.
                // sockets.erase(it);
            }
        }
    }
    broadcast(make_message(sock, "Leave!"));
}
