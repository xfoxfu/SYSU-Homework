#include "socket.hpp"

using namespace std;

int checked_run(int ret) {
  if (ret == SOCKET_ERROR) {
    ostringstream iss;
    iss << "Operation failed with reason: " << strerror(errno);
    throw logic_error(iss.str());
  }
  return ret;
}

void fox_socket::_ensure_connection() const {
  if (!_connected) {
    throw new logic_error("not connected");
  }
}

void fox_socket::_ensure_no_connection() const {
  if (_connected) {
    throw new logic_error("already connected");
  }
}

fox_socket::fox_socket(const char *ip, const char *port)
    : fox_socket(inet_addr(ip), htons(atoi(port))) {}

fox_socket::fox_socket(uint32_t ip, uint16_t port) {
  _sock_id = ::socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
  _addr.sin_family = AF_INET; // 因特网地址簇(INET-Internet)
  _addr.sin_addr.s_addr = ip; // 监听所有(接口的)IP地址。
  _addr.sin_port = port;
  _connected = false;
}

fox_socket::fox_socket(int sock_id, struct sockaddr_in addr) {
  _sock_id = sock_id;
  _addr = addr;
  _connected = true;
}

void fox_socket::bind() {
  _ensure_no_connection();
  checked_run(::bind(_sock_id, (struct sockaddr *)&_addr, sizeof(_addr)));
  checked_run(::listen(_sock_id, 5));
  _connected = true;
}

void fox_socket::connect() {
  _ensure_no_connection();
  checked_run(::connect(_sock_id, reinterpret_cast<struct sockaddr *>(&_addr),
                        sizeof(_addr)));
  _connected = true;
}

fox_socket fox_socket::accept() {
  _ensure_connection();
  struct sockaddr_in remote;
  auto remote_size = static_cast<socklen_t>(sizeof(remote));
  auto rsock_id = checked_run(::accept(
      _sock_id, reinterpret_cast<struct sockaddr *>(&remote), &remote_size));
  return fox_socket(rsock_id, remote);
}

void fox_socket::close() {
  if (_connected) {
    checked_run(::close(_sock_id));
    _connected = false;
  }
}

void fox_socket::send(const string &str) const {
  checked_run(::send(_sock_id, str.c_str(), str.length(), 0));
}

string fox_socket::recv() {
  size_t has_received = 0;
  char buf[BUF_LEN] = {0};
  ostringstream os;
  auto ret = ::recv(_sock_id, buf, BUF_LEN, 0);
  checked_run(ret);
  if (ret == 0) {
    _connected = false;
    ostringstream iss;
    iss << "Remote disconnected, sock_id=" << _sock_id;
    throw logic_error(iss.str());
  }
  has_received += ret;
  os.write(buf, ret);
  return os.str();
}

uint32_t fox_socket::ip() const { return _addr.sin_addr.s_addr; }

uint16_t fox_socket::port() const { return _addr.sin_port; }

bool fox_socket::operator==(const fox_socket &rhs) const noexcept {
  return _sock_id == rhs._sock_id;
}

bool fox_socket::connected() const noexcept { return _connected; }

int fox_socket::sock_id() const { return _sock_id; }
