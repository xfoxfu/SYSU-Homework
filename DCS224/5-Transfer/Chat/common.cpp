#include "common.hpp"
#include <cassert>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

using namespace std;

string make_filename(const string &dir, const string &orig, size_t nonce) {
  fs::path path(dir);
  path /= orig;
  path.replace_filename(path.stem().string() +
                        (nonce > 1 ? ("(" + to_string(nonce) + ")") : "") +
                        path.extension().string());
  return path;
}

string try_filename(const string &directory, const string &filename) {
  size_t suffix = 0;
  string ret;
  while (fs::exists(ret = make_filename(directory, filename, suffix))) {
    suffix += 1;
  }

  return ret;
}

void send_chat(fox_socket &sock, const string &content) {
  transfer_head head;
  head.type = transfer_type::chat;
  head.length1 = content.length();
  head.length2 = 0;

  sock.send(head);
  sock.send(content);
}

void send_file(fox_socket &sock, const string &dir, const string &filename) {
  transfer_head head;
  head.type = transfer_type::file;
  head.length1 = filename.length();

  ostringstream buf;
  ifstream fin(make_filename(dir, filename, 0));
  buf << fin.rdbuf();

  string content = buf.str();

  head.length2 = content.length();
  sock.send(head);
  sock.send(filename);
  sock.send(content);
}

void recv_misc(fox_socket &sock, const string &dir) {
  transfer_head head;
  sock.recv(head);
  if (head.type == transfer_type::chat) {
    recv_chat(sock, head);
  } else if (head.type == transfer_type::file) {
    recv_file(sock, head, dir);
  } else {
    assert(false);
  }
}

void recv_chat(fox_socket &sock, const transfer_head &head) {
  auto data = sock.recv(static_cast<size_t>(head.length1));
  cout << '\r' << data << endl << "> " << flush;
}

void recv_file(fox_socket &sock, const transfer_head &head, const string &dir) {
  auto filename = sock.recv(static_cast<size_t>(head.length1));
  auto content = sock.recv(static_cast<size_t>(head.length2));
  cout << '\r' << "Received: " << filename << endl << "> " << flush;

  auto dest = try_filename(dir, filename);
  ofstream fout(dest);
  istringstream is(content);
  fout << is.rdbuf();
}
