// FileUnpack.cpp
#include "FileStruct.hpp"
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::ifstream;
using std::map;
using std::ofstream;
using std::string;
namespace fs = std::filesystem;

void read_str(const char *prompt, string &str) {
  cout << prompt;
  getline(cin, str);
}

string make_filename(const string &dir, const string &orig, size_t nonce) {
  fs::path path(dir);
  path /= orig;
  path.replace_filename(path.stem().string() +
                        (nonce > 1 ? ("(" + std::to_string(nonce) + ")") : "") +
                        path.extension().string());
  return path;
}

int main() {
  string pakname;
  cout << "Filename: ";
  getline(cin, pakname);

  string directory;
  cout << "Target Dir: ";
  getline(cin, directory);
  fs::create_directory(directory);

  ifstream fin(pakname, std::ios::binary);
  while (!fin.eof()) {
    FileStruct filedata;
    fin.read(reinterpret_cast<char *>(&filedata), sizeof(filedata));
    if (fin.eof()) {
      break;
    }

    size_t suffix = 0;
    while (fs::exists(make_filename(
        directory, string(reinterpret_cast<const char *>(filedata.fileName)),
        suffix))) {
      suffix += 1;
    }

    string filename = make_filename(
        directory, string(reinterpret_cast<const char *>(filedata.fileName)),
        suffix);

    uint8_t *buf = new uint8_t[filedata.fileSize];
    fin.read(reinterpret_cast<char *>(buf), filedata.fileSize);

    ofstream fout(filename, std::ios::binary);
    fout.write(reinterpret_cast<const char *>(buf), filedata.fileSize);

    delete[] buf;
  }
  return 0;
}
