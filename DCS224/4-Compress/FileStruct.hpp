// FileStruct.hpp
#include <cstdint>

typedef struct FileStruct {
  int8_t fileName[300];
  uint64_t fileSize;
} FileStruct;
