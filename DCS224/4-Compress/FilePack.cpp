// FilePack.cpp
#include "FileStruct.hpp"
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

void read_str(const char *prompt, string &str) {
  cout << prompt;
  getline(cin, str);
}

int main() {
  string filename;
  cout << "Filename: ";
  getline(cin, filename);

  ofstream fout(filename, std::ios::binary);
  size_t count = 0;
  while (
      read_str(("File #" + std::to_string(count++) + ": ").c_str(), filename),
      filename != "exit") {
    FileStruct filedata;
    memset(reinterpret_cast<void *>(&filedata), 0, sizeof(filedata));
    strcpy(reinterpret_cast<char *>(filedata.fileName), filename.c_str());

    ifstream file(filename, std::ios::in | std::ios::binary);
    file.seekg(0, file.end);
    size_t length = file.tellg();
    file.seekg(0, file.beg);

    filedata.fileSize = length;
    fout.write(reinterpret_cast<const char *>(&filedata), sizeof(filedata));

    fout << file.rdbuf();
  }
  return 0;
}
