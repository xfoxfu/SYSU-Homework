#include "biosfn.h"
#include <stdint.h>

void run(uint8_t id) {
  int8_t disc = syscall_get_default_drive();
  load_sector(0x0A00, 0x0100, 0x00, id * 3 + 8);
  load_sector(0x0A00, 0x0300, 0x00, id * 3 + 9);
  load_sector(0x0A00, 0x0500, 0x00, id * 3 + 10);
  syscall_far_jump_A00();
}
void kmain(void) {
  char prompt_command[] = "Command: ";
  print_str(prompt_command);

  for (;;) {
    // read command
    int8_t command[80];
    uint8_t count = 0;
    syscall_set_cursor_type(0b00001111, 0b00001110);
    for (; count < 80;) {
      int8_t ch = syscall_get_key_block();
      if (ch != '\n' && ch != '\r') {
        syscall_put_char(ch);
        command[count] = ch;
        count = count + 1;
      } else {
        syscall_put_char('\n');
        syscall_put_char('\r');
        break;
      }
    }

    // execute command
    for (int i = 0; i < count; i++) {
      run(command[i] - '0');
    }
  }
}
