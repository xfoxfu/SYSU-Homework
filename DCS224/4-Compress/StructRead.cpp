// StructRead.cpp
#include "Person.hpp"
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;

void print_time(const char *prompt, uint32_t time) {
  time_t tm = time;
  cout << prompt << std::put_time(std::localtime(&tm), "%c %Z") << endl;
}

int main() {
  ifstream fin("Persons.stru");
  while (!fin.eof()) {
    Person person;
    fin.read(reinterpret_cast<char *>(&person), sizeof(person));
    if (fin.eof()) {
      break;
    }
    cout << "Name:  " << person.username << endl;
    cout << "Level: " << person.level << endl;
    cout << "Email: " << person.email << endl;
    print_time("Send:  ", person.sendtime);
    print_time("Reg:   ", person.regtime);
  }
  return 0;
}
