#include "biosfn.h"
#include <stdint.h>

const char *prompt_command = "Command: ";
const char *help = "Help:\n\r"
                   "Number sequence - run commands\n\r"
                   "h - help\n\r"
                   "x - exit\n\r"
                   "l - list commands\n\r";
const char *error_occurred = "Error occurred when loading binary: ";

void run(uint8_t id) {
  int8_t disc = syscall_get_default_drive();
  int8_t err = load_sector(0x0A00, 0x0100, 0x00, id * 3 + 8);
  if (err) {
    print_str(error_occurred);
    syscall_put_char('A');
    print_u8_hex(err);
    print_str("\n\r");
    return;
  }
  syscall_far_jump_A00();
}
void kmain(void) {
  for (;;) {
    print_str(prompt_command);
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
    command[count] = '\0';

    for (int i = 0; i < count; i++) {
      if (command[i] == 'h') {
        print_str(help);
      } else if (command[i] == 'x') {
        return;
      } else {
        run(command[i] - '0');
      }
    }
  }
}
