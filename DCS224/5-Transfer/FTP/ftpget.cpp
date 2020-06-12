#include "socket.hpp"
#include "socket_stream.hpp"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <queue>

using namespace std;

string send_command(tcp_stream &sock, const string &command);
string recv_reply(tcp_stream &sock);

string send_command(tcp_stream &sock, const string &command) {
  sock << command << "\r\n" << flush;
  cout << command << endl;
  return recv_reply(sock);
}

string recv_reply(tcp_stream &sock) {
  try {
    string s;
    getline(sock, s);
    cout << s << endl;
    return s;
  } catch (const std::exception &ex) {
    if (!(strstr(ex.what(), "Bad file descriptor") != nullptr)) {
      throw ex;
    }
    return "";
  }
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    cout << "Usage: " << argv[0] << " server remote_file local_file" << endl;
    return 1;
  }
  const char *server = argv[1];
  const char *remote = argv[2];
  const char *local = argv[3];

  cout << "[INFO] Trying " << server << ":21" << endl;
  fox_socket ctrl(server, "21");
  ctrl.connect();
  auto ctrl_s = tcp_stream(ctrl.sock_id(), 100);

  recv_reply(ctrl_s);

  send_command(ctrl_s, "user net");
  send_command(ctrl_s, "pass 123456");
  auto data_addr_str = send_command(ctrl_s, "pasv");
  int code, ip_a, ip_b, ip_c, ip_d, port_a, port_b;
  sscanf(data_addr_str.c_str(), "%d Entering Passive Mode (%d,%d,%d,%d,%d,%d).",
         &code, &ip_a, &ip_b, &ip_c, &ip_d, &port_a, &port_b);
  cout << "[INFO] Trying " << server << ":" << (port_a * 256 + port_b) << endl;

  fox_socket data(server, to_string(port_a * 256 + port_b).c_str());
  tcp_stream data_s(data.sock_id(), 1000);
  data.connect();

  send_command(ctrl_s, string("retr ") + remote);
  try {
    ofstream fout(local);
    fout << data_s.rdbuf();
  } catch (...) {
    cout << "[INFO] Data connection closed" << endl;
  }
  recv_reply(ctrl_s);
  send_command(ctrl_s, "quit");

  return 0;
}
