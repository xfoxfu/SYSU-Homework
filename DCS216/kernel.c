#include "biosfn.h"
#include <stdint.h>

void run(uint8_t id) {
  int8_t disc = syscall_get_default_drive();
  load_sector(0x0A00, 0x0100, disc, id * 3 + 8);
  load_sector(0x0A00, 0x0300, disc, id * 3 + 9);
  load_sector(0x0A00, 0x0500, disc, id * 3 + 10);
  syscall_far_jump_A00();
}
void kmain(void) {
  syscall_set_cursor_type(0b0001111, 0b00100000);
  run(0);
  run(1);
  run(2);
  run(3);
}
