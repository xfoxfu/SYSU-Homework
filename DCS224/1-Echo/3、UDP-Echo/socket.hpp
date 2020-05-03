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
    struct sockaddr_in _self_addr;
    struct sockaddr_in _remote_addr;

public:
    fox_socket(uint32_t ip, uint16_t port);
    fox_socket(int sock_id, struct sockaddr_in addr);
    void bind();
    void close();
    void send(const std::string &str);
    std::pair<std::string, struct sockaddr_in> recv();
    uint32_t ip();
    uint16_t port();
    uint32_t remote_ip();
    uint16_t remote_port();
};

fox_socket::fox_socket(uint32_t ip, uint16_t port)
{
    _sock_id = ::socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
    _self_addr.sin_family = AF_INET;
    _self_addr.sin_addr.s_addr = ip;
    _self_addr.sin_port = port;
    _remote_addr = _self_addr;
}

fox_socket::fox_socket(int sock_id, struct sockaddr_in addr)
{
    _sock_id = sock_id;
    _remote_addr = addr;
}

void fox_socket::bind()
{
    checked_run(::bind(_sock_id, (struct sockaddr *)&_self_addr, sizeof(_self_addr)));
}

void fox_socket::close()
{
    checked_run(::close(_sock_id));
}

void fox_socket::send(const std::string &str)
{
    checked_run(::sendto(_sock_id, str.c_str(), str.length(), 0,
                         reinterpret_cast<const struct sockaddr *>(&_remote_addr), sizeof(_remote_addr)));
}

std::pair<std::string, struct sockaddr_in> fox_socket::recv()
{
    bool has_received = 0;
    char buf[BUF_LEN] = {0};
    std::ostringstream os;
    do
    {
        auto size_addr = static_cast<socklen_t>(sizeof(_remote_addr));
        auto ret = ::recvfrom(_sock_id, buf, BUF_LEN, MSG_DONTWAIT,
                              reinterpret_cast<struct sockaddr *>(&_remote_addr), &size_addr);
        if (ret == SOCKET_ERROR && (errno == EAGAIN || errno == EWOULDBLOCK))
        {
            // ignore the error
        }
        else
        {
            checked_run(ret);
            os.write(buf, ret);
            has_received = true;
        }
    } while (has_received != true);
    return make_pair(os.str(), _remote_addr);
}

uint32_t fox_socket::ip()
{
    return _self_addr.sin_addr.s_addr;
}

uint16_t fox_socket::port()
{
    return _self_addr.sin_port;
}

uint32_t fox_socket::remote_ip()
{
    return _remote_addr.sin_addr.s_addr;
}

uint16_t fox_socket::remote_port()
{
    return _remote_addr.sin_port;
}
