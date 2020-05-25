#include "Person.hpp"
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;

void read_str(const char *prompt, string &str) {
  cout << prompt;
  getline(cin, str);
}

void read_int32(const char *prompt, int32_t &val) {
  cout << prompt;
  string str;
  getline(cin, str);
  val = atoi(str.c_str());
}

int main() {
  string name;
  int32_t level;
  string email;

  ofstream fout("Persons.stru");
  while (read_str("Name: ", name), name != "exit") {
    read_int32("Level: ", level);
    read_str("Email: ", email);

    Person person;
    memset(reinterpret_cast<void *>(&person), 0, sizeof(person));
    strcpy(reinterpret_cast<char *>(person.username), name.c_str());
    person.level = level;
    strcpy(reinterpret_cast<char *>(person.email), email.c_str());
    time_t now;
    std::time(&now);
    person.regtime = now;
    person.sendtime = now;

    fout.write(reinterpret_cast<const char *>(&person), sizeof(person));
  }
  return 0;
}
