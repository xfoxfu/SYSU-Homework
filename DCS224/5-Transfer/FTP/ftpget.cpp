#include "socket.hpp"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <queue>

using namespace std;

string send_command(fox_socket &sock, const string &command);
string recv_reply(fox_socket &sock);

string send_command(fox_socket &sock, const string &command) {
  sock.send(command);
  sock.send("\r\n");
  cout << command << endl;
  return recv_reply(sock);
}

string recv_reply(fox_socket &sock) {
  try {
    auto ret = sock.recv();
    cout << ret;
    return ret;
  } catch (const std::exception &ex) {
    if (!(strstr(ex.what(), "Bad file descriptor") != nullptr)) {
      throw ex;
    }
    return "";
  }
}

int main(int argc, char *argv[]) {
  const char *server = argv[1];
  const char *remote = argv[2];
  const char *local = argv[3];

  cout << "Trying " << server << ":21" << endl;
  fox_socket ctrl(server, "21");
  ctrl.connect();

  recv_reply(ctrl);

  send_command(ctrl, "user net");
  send_command(ctrl, "pass 123456");
  auto data_addr_str = send_command(ctrl, "pasv");
  int code, ip_a, ip_b, ip_c, ip_d, port_a, port_b;
  sscanf(data_addr_str.c_str(), "%d Entering Passive Mode (%d,%d,%d,%d,%d,%d).",
         &code, &ip_a, &ip_b, &ip_c, &ip_d, &port_a, &port_b);
  cout << "Trying " << server << ":" << (port_a * 256 + port_b) << endl;

  fox_socket data(server, to_string(port_a * 256 + port_b).c_str());
  data.connect();

  send_command(ctrl, string("retr ") + remote);
  try {
    ofstream fout(local);
    while (true) {
      auto part = data.recv();
      fout.write(part.c_str(), part.length());
    }
  } catch (...) {
    cout << "Data connection closed" << endl;
  }
  recv_reply(ctrl);
  send_command(ctrl, "quit");

  return 0;
}
