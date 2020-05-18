#include "socket.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::thread;
using std::chrono::system_clock;

const char *PROMPT = ">> ";

void handle(fox_socket sock);

int main(int argc, char *argv[]) {
  const char *host = "127.0.0.1";
  const char *port = "50500";
  //   cout << "Usage: " << argv[0] << " [server=" << host << "] [port=" << port
  //        << "]" << endl;
  if (argc >= 2)
    host = argv[1];
  if (argc >= 3)
    port = argv[2];
  fox_socket sock(host, port);
  try {
    sock.connect();

    auto th = thread(handle, sock);

    while (sock.connected()) {
      string msg;
      getline(cin, msg);

      sock.send(msg);
      sock.send("\r\n");
    }
  } catch (std::exception &e) {
    cout << e.what() << endl;
  }
}

void handle(fox_socket sock) {
  try {
    while (true) {
      auto str = sock.recv();
      cout << str << std::flush;
    }
  } catch (const std::exception &ex) {
    if (!(strstr(ex.what(), "Bad file descriptor") != nullptr)) {
      cout << ex.what() << endl;
      exit(0);
    }
  }
}
