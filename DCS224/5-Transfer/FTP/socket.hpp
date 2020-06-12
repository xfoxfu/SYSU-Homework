#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <sstream>
#include <stdexcept>
#include <sys/errno.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef __MACH__
#include <mach/error.h>
#else
#include <error.h>
#endif

#define SOCKET_ERROR -1
#define BUF_LEN 100

class fox_socket {
  int _sock_id;
  struct sockaddr_in _addr;
  bool _connected;

  void _ensure_connection() const;
  void _ensure_no_connection() const;

public:
  fox_socket(const char *ip, const char *port);
  fox_socket(uint32_t ip, uint16_t port);
  fox_socket(int sock_id, struct sockaddr_in addr);
  void bind();
  fox_socket accept();
  void connect();
  void close();
  void send(const std::string &str) const;
  std::string recv();
  uint32_t ip() const;
  uint16_t port() const;
  bool operator==(const fox_socket &rhs) const noexcept;
  bool connected() const noexcept;
  int sock_id() const;
};
