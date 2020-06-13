#pragma once

#include "socket.hpp"
#include "transfer.hpp"
#include <iostream>
#include <string>

std::string make_filename(const std::string &dir, const std::string &orig,
                          size_t nonce);
std::string try_filename(const std::string &directory,
                         const std::string &filename);
void send_chat(fox_socket &sock, const std::string &content);
void send_file(fox_socket &sock, const std::string &dir,
               const std::string &filename);
void recv_misc(fox_socket &sock, const std::string &dir);
void recv_chat(fox_socket &sock, const transfer_head &head);
void recv_file(fox_socket &sock, const transfer_head &head,
               const std::string &dir);
