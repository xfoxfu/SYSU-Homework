#include "biosfn.h"
#include <stdint.h>

void run(uint8_t id) {
  int8_t disc = syscall_get_default_drive();
  syscall_load_sector(0x0A00, 0x0100, disc, id * 3 + 8);
  print_u8_hex(syscall_status_last_op(disc));
  syscall_load_sector(0x0A00, 0x0300, disc, id * 3 + 9);
  print_u8_hex(syscall_status_last_op(disc));
  syscall_load_sector(0x0A00, 0x0500, disc, id * 3 + 10);
  print_u8_hex(syscall_status_last_op(disc));
  syscall_put_char('>');
  syscall_far_jump_A00();
}
void kmain(void) {
  syscall_put_char('#');
  run(0);
  syscall_put_char('#');
  run(1);
  syscall_put_char('#');
  run(2);
  syscall_put_char('#');
  run(3);
}
