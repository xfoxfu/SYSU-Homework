#include "biosfn.h"
#include <stdint.h>

const char *prompt_command = "xfoxos> ";
const char *help = "==================\n\r"
                   "Help:\n\r"
                   "h - help\n\r"
                   "x - exit\n\r"
                   "l - list commands\n\r"
                   "c - clear screen\n\r"
                   "run commands:\n\r";
const char *error_occurred = "Error occurred when loading binary: ";
const char *msg_is = " => ";
const char *invalid_exec = "Invalid executable. \n\r";

uint8_t count_executables() {
  int8_t disc = syscall_get_default_drive();
  uint8_t err = load_sector(0x0A00, 0x0000, 0x00, 10, 1);
  if (err) {
    print_str(error_occurred);
    print_u8_hex(err);
    print_str("\n\r");
    return 0;
  }
  uint8_t count = 0;
  for (uint8_t i = 0; i < 0x0200; i++) {
    print_u8_hex(count);
    print_str(msg_is);
    int8_t value;
    do {
      value = raw_far_pointer_read(0x0A00, i);
      syscall_put_char(value);
      i++;
    } while (value != '\n' && value != '\r' && value != '\0');
    syscall_put_char('\n');
    syscall_put_char('\r');
    count++;
    if (value == '\0')
      break;
  }
  return count;
}

uint8_t print_help() {
  print_str(help);
  return count_executables();
}

void run(uint8_t id) {
  int8_t disc = syscall_get_default_drive();
  uint8_t err = load_sector(0x0A00, 0x0100, 0x00, id * 3 + 11, 3);
  if (err) {
    print_str(error_occurred);
    print_u8_hex(err);
    print_str("\n\r");
    return;
  }
  syscall_far_jump_A00();
}
void kmain(void) {
  uint8_t execs = print_help();
  for (;;) {
    print_str(prompt_command);
    // read command
    int8_t command[80];
    uint8_t count = 0;
    syscall_set_cursor_type(0b00001111, 0b00001110);
    for (; count < 80;) {
      int8_t ch = syscall_get_key_block();
      if (ch == '\b') {
        syscall_put_char(ch);
        syscall_put_char(' ');
        syscall_put_char(ch);
        count -= 1;
      } else if (ch != '\n' && ch != '\r') {
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
        print_help();
      } else if (command[i] == 'l') {
        count_executables();
      } else if (command[i] == 'x') {
        return;
      } else if (command[i] == 'c') {
        syscall_clear_screen();
      } else {
        if (command[i] >= execs + '0') {
          print_str(invalid_exec);
          continue;
        }
        run(command[i] - '0');
      }
    }
  }
}
