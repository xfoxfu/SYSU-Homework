#include "common.hpp"
#include "socket.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

using namespace std;

mutex dir_lock;
string directory;

const char *PROMPT = "> ";

void handler(fox_socket sock);

int main(int argc, char *argv[]) {
  const char *host = "127.0.0.1";
  const char *port = "50500";
  cout << "Usage: " << argv[0] << " [server=" << host << "] [port=" << port
       << "]" << endl;
  if (argc >= 2)
    host = argv[1];
  if (argc >= 3)
    port = argv[2];
  cout << "Connecting to " << host << ":" << port << endl;
  fox_socket sock(host, port);
  try {
    sock.connect();
    cout << "Connected." << endl;

    auto th = thread(handler, sock);

    while (sock.connected()) {
      string msg;
      cout << PROMPT << std::flush;
      getline(cin, msg);

      if (msg == "exit") {
        cout << "Leave!" << endl;
        sock.close();
        th.join();
        break;
      } else if (msg.find("rdir") == 0) {
        {
          auto _guard = std::lock_guard(dir_lock);
          directory = msg.substr(5);
        }
      } else if (msg.find("send") == 0) {
        {
          auto _guard = std::lock_guard(dir_lock);
          send_file(sock, directory, msg.substr(5));
        }
      } else {
        send_chat(sock, msg);
      }
    }
  } catch (std::exception &e) {
    cout << e.what() << endl;
  }
}

void handler(fox_socket sock) {
  while (sock.connected()) {
    try {
      { recv_misc(sock, directory); }
    } catch (const std::exception &ex) {
      cout << "Error encountered while recv: " << ex.what() << endl;
      if (strstr(ex.what(), "Remote disconnected") != nullptr ||
          strstr(ex.what(), "Bad file descriptor") != nullptr) {
        exit(0);
      }
    }
  }
}
