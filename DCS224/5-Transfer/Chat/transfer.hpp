#include <cstdint>

enum class transfer_type : uint8_t { chat, file };

struct transfer_head {
  transfer_type type;
  uint64_t length1;
  uint64_t length2;
};
