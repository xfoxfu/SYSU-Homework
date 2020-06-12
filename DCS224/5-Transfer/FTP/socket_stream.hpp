#pragma once

#include <iostream>

class tcp_streambuf : public std::streambuf {
public:
  tcp_streambuf(int socket, size_t buf_size);
  ~tcp_streambuf();

  int underflow();
  int overflow(int c);
  int sync();

private:
  const size_t buf_size_;
  int socket_;
  char *pbuf_;
  char *gbuf_;
};

class tcp_stream : public std::iostream {
public:
  tcp_stream(int socket, size_t buf_size);
  ~tcp_stream();

private:
  int socket_;
  const size_t buf_size_;
};
