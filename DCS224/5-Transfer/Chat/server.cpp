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
  const char *port = "50500";
  cout << "Usage: " << argv[0] << " [port=" << port << "]" << endl;
  if (argc >= 2) {
    port = argv[1];
  }
  cout << "Binding on 0.0.0.0:" << port << endl;
  fox_socket sock("0.0.0.0", port);
  try {
    sock.bind();
    cout << "Server started." << endl << endl;

    auto sub_sock = sock.accept();
    cout << "Connected." << endl;

    auto th = thread(handler, sub_sock);

    while (sub_sock.connected()) {
      string msg;
      cout << PROMPT << std::flush;
      getline(cin, msg);

      if (msg == "exit") {
        cout << "Leave!" << endl;
        sub_sock.close();
        th.join();
        break;
      } else if (msg.rfind("rdir", 0) == 0) {
        {
          auto _guard = std::lock_guard(dir_lock);
          directory = msg.substr(5);
        }
      } else if (msg.rfind("send", 0) == 0) {
        {
          auto _guard = std::lock_guard(dir_lock);
          send_file(sub_sock, directory, msg.substr(5));
        }
      } else {
        send_chat(sub_sock, msg);
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
