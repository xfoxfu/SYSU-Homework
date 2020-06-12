
#include "socket_stream.hpp"
#include <assert.h>
#include <stddef.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace std;

tcp_streambuf::tcp_streambuf(int socket, size_t buf_size)
    : buf_size_(buf_size), socket_(socket) {
  assert(buf_size_ > 0);
  pbuf_ = new char[buf_size_];
  gbuf_ = new char[buf_size_];

  setp(pbuf_, pbuf_ + buf_size_);
  setg(gbuf_, gbuf_, gbuf_);
}

tcp_streambuf::~tcp_streambuf() {
  if (pbuf_ != nullptr) {
    delete pbuf_;
    pbuf_ = nullptr;
  }

  if (gbuf_ != nullptr) {
    delete gbuf_;
    gbuf_ = nullptr;
  }
}

int tcp_streambuf::sync() {
  int sent = 0;
  int total = pptr() - pbase();
  while (sent < total) {
    int ret = send(socket_, pbase() + sent, total - sent, 0);
    if (ret > 0)
      sent += ret;
    else {
      return -1;
    }
  }
  setp(pbase(), pbase() + buf_size_);
  pbump(0);

  return 0;
}

int tcp_streambuf::overflow(int c) {
  if (-1 == sync()) {
    return traits_type::eof();
  } else {
    if (!traits_type::eq_int_type(c, traits_type::eof())) {
      sputc(traits_type::to_char_type(c));
    }

    return traits_type::not_eof(c);
  }
}

int tcp_streambuf::underflow() {
  int ret = recv(socket_, eback(), buf_size_, 0);
  if (ret > 0) {
    setg(eback(), eback(), eback() + ret);
    return traits_type::to_int_type(*gptr());
  } else {
    return traits_type::eof();
  }
  return 1;
}

tcp_stream::tcp_stream(int socket, size_t buf_size)
    : iostream(new tcp_streambuf(socket, buf_size)), socket_(socket),
      buf_size_(buf_size) {}

tcp_stream::~tcp_stream() { delete rdbuf(); }
