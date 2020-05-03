#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <sys/errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#include <stdexcept>
#include <sstream>

#ifdef __MACH__
#include <mach/error.h>
#else
#include <error.h>
#endif

#define SOCKET_ERROR -1
#define BUF_LEN 100

int checked_run(int ret)
{
    if (ret == SOCKET_ERROR)
    {
        std::ostringstream iss;
        iss << "Operation failed with reason: " << strerror(errno);
        throw std::logic_error(iss.str());
    }
    return ret;
}

class fox_socket
{
    int _sock_id;
    struct sockaddr_in _addr;
    bool _connected;
    void _ensure_connection() const;
    void _ensure_no_connection() const;

public:
    fox_socket(uint32_t ip, uint16_t port);
    fox_socket(int sock_id, struct sockaddr_in addr);
    void bind();
    fox_socket accept();
    void connect();
    void close();
    void send(const std::string &str) const;
    std::string recv();
    uint32_t ip();
    uint16_t port();
};

void fox_socket::_ensure_connection() const
{
    if (!_connected)
    {
        throw new std::logic_error("not connected");
    }
}

void fox_socket::_ensure_no_connection() const
{
    if (_connected)
    {
        throw new std::logic_error("already connected");
    }
}

fox_socket::fox_socket(uint32_t ip, uint16_t port)
{
    _sock_id = ::socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
    _addr.sin_family = AF_INET; // 因特网地址簇(INET-Internet)
    _addr.sin_addr.s_addr = ip; // 监听所有(接口的)IP地址。
    _addr.sin_port = port;
    _connected = false;
}

fox_socket::fox_socket(int sock_id, struct sockaddr_in addr)
{
    _sock_id = sock_id;
    _addr = addr;
    _connected = true;
}

void fox_socket::bind()
{
    _ensure_no_connection();
    checked_run(::bind(_sock_id, (struct sockaddr *)&_addr, sizeof(_addr)));
    checked_run(::listen(_sock_id, 5));
    _connected = true;
}

void fox_socket::connect()
{
    _ensure_no_connection();
    checked_run(::connect(_sock_id, reinterpret_cast<struct sockaddr *>(&_addr), sizeof(_addr)));
    _connected = true;
}

fox_socket fox_socket::accept()
{
    _ensure_connection();
    struct sockaddr_in remote;
    auto remote_size = static_cast<socklen_t>(sizeof(remote));
    auto rsock_id = checked_run(::accept(_sock_id, reinterpret_cast<struct sockaddr *>(&remote), &remote_size));
    return fox_socket(rsock_id, remote);
}

void fox_socket::close()
{
    checked_run(::close(_sock_id));
}

void fox_socket::send(const std::string &str) const
{
    // std::cout << "PRINTED=" << str.length() << std::endl;
    checked_run(::send(_sock_id, str.c_str(), str.length() + 1, 0));
}

std::string fox_socket::recv()
{
    size_t has_received = 0;
    char buf[BUF_LEN] = {0};
    std::ostringstream os;
    auto ret = ::recv(_sock_id, buf, BUF_LEN, 0);
    checked_run(ret);
    if (ret == 0)
    {
        throw std::logic_error("Remote disconnected");
    }
    has_received += ret;
    os.write(buf, ret);
    return os.str();
}

uint32_t fox_socket::ip()
{
    return _addr.sin_addr.s_addr;
}

uint16_t fox_socket::port()
{
    return _addr.sin_port;
}
