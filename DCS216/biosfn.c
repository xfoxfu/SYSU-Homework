#include "biosfn.h"

void raw_far_pointer_write(uint16_t segment, uint16_t offset, int8_t value) {
  asm("pushw es        \n"
      "mov es, %1      \n"
      "mov bx, %2      \n"
      "mov byte ptr es:[bx], %0 \n"
      "popw es         \n"
      : // no output
      : "r"(value), "g"(segment), "g"(offset)
      : "bx");
}

int8_t raw_far_pointer_read(uint16_t segment, uint16_t offset) {
  int8_t ret;
  asm("pushw es        \n"
      "mov es, %1      \n"
      "mov bx, %2      \n"
      "mov %0, byte ptr es:[bx] \n"
      "popw es         \n"
      : "=r"(ret)
      : "g"(segment), "g"(offset)
      : "bx");
  return ret;
}

void syscall_display_set_char(int16_t row, int16_t col, int8_t chr,
                              uint8_t color) {
  raw_far_pointer_write(0xB800, (row * 80 + col) * 2, chr);
  raw_far_pointer_write(0xB800, (row * 80 + col) * 2 + 1, color);
}

void syscall_sleep(int16_t time_ms) {
  asm volatile("mov cx, 0x0000 \n"
               "mov ah, 0x86   \n"
               "mov al, 0x00   \n"
               "int 0x15       \n"
               : /* no output */
               : "d"(time_ms)
               : "cx", "ax");
}

void syscall_display_get_char(int16_t row, int16_t col, int8_t *chr,
                              int8_t *color) {
  *chr = raw_far_pointer_read(0xB800, 2 * (row * 80 + col));
  *color = raw_far_pointer_read(0xB800, 2 * (row * 80 + col) + 1);
}

void display(const int8_t x, const int8_t y, const int8_t xm, const int8_t ym) {
  syscall_set_cursor_type(0b00000000, 0b00100000);
  char NAME[] = {'1', '7', '3', '4', '1', '0', '3', '9'};
  int col = x;
  int row = y;
  int col_incr = 1;
  int row_incr = 1;
  int color = VGA_White;
  char protect[8 * 2] = {0};

  for (;;) {
    syscall_display_set_char(row, col, '*', color);
    for (int i = 0; i < 8; i++) {
      syscall_display_get_char(row, col + 1 + i, &protect[2 * i],
                               &protect[2 * i + 1]);
    }
    for (int i = 0; i < 8; i++) {
      if (col + 1 + i >= xm) {
        break;
      }
      syscall_display_set_char(row, col + 1 + i, NAME[i], color);
    }

    syscall_sleep(25000);

    for (int i = 0; i < 8; i++) {
      if (col + 1 + i >= xm) {
        break;
      }

      syscall_display_set_char(row, col + 1 + i, protect[2 * i],
                               protect[2 * i + 1]);
    }

    col += col_incr;
    row += row_incr;
    if (col == x)
      col_incr = 1;
    if (col == xm - 1)
      col_incr = -1;
    if (row == y)
      row_incr = 1;
    if (row == ym - 1)
      row_incr = -1;

    if (col == x || col == xm - 1 || row == y || row == ym - 1) {
      color += 1;
    }
    if (color > VGA_White)
      color = VGA_Blue;

    if (syscall_get_key_noblock() == 'x') {
      return;
    }
  }
}

int8_t syscall_get_key_noblock(void) {
  int8_t key;
  asm("   mov ah, 0x01  \n"
      "   int 0x16      \n"
      "   jz  1f        \n"
      "   mov ah, 0x00  \n"
      "   int 0x16      \n"
      "   mov %0, al    \n"
      "   jmp 2f        \n"
      "1: mov %0, 0x00  \n"
      "2:               \n"
      : "=r"(key)
      : /* no input */
      : "ah", "al");
  return key;
}

int8_t syscall_get_key_block(void) {
  int8_t key;
  asm("mov ah, 0x00 \n"
      "int 0x16     \n"
      "mov %0, al   \n"
      : "=r"(key)
      : /* no input */
      : "ah", "al");
  return key;
}

int8_t syscall_get_default_drive(void) {
  int8_t drive;
  asm("mov ah, 0x0E \n"
      "int 0x21     \n"
      "mov %0, dl   \n"
      : "=r"(drive)
      : /* no input */
      : "ah", "dl");
  return drive;
}

int8_t syscall_status_last_op(int8_t disc) {
  int8_t status;
  asm("mov ah, 0x01 \n"
      "mov dl, %1   \n"
      "int 0x13     \n"
      "mov %0, ah   \n"
      : "=r"(status)
      : "g"(disc)
      : "ah");
  return status;
}

void syscall_put_char(int8_t ch) {
  asm("mov ah, 0x0E   \n"
      "mov al, %0     \n"
      "mov bx, 0x0000 \n"
      "int 0x10       \n"
      : /* no output */
      : "g"(ch)
      : "ax", "bx");
}

void syscall_load_sector(int16_t segment, int16_t offset, int8_t disc,
                         int8_t sector, uint8_t length) {
  asm volatile("pushad         \n" // too many reg used, protect them all
               "pushw es       \n"
               "mov   bx, %0   \n"
               "mov   es, bx   \n"
               "mov   bx, %1   \n"
               "mov   ah, 0x02 \n"
               "mov   al, %4   \n"
               "mov   dh, 0x00 \n"
               "mov   dl, %2   \n"
               "mov   ch, 0x00 \n"
               "mov   cl, %3   \n"
               "int   0x13     \n"
               "popw  es       \n"
               "popad          \n"
               : /* no output */
               : "m"(segment), "m"(offset), "m"(disc), "m"(sector), "m"(length)
               // cannot put these as registers
               : "ax", "bx", "cx", "dx", "memory");
}

void syscall_far_jump_A00() {
  asm volatile("pusha              \n"
               "callw 0x0A00:0x0100\n"
               "popa               \n"
               "mov   ax, cs       \n"
               "mov   ds, ax       \n"
               "mov   es, ax       \n"
               : // no output
               : // no input
               : "ax");
}

void print_u8_hex(uint8_t num) {
  if ((num / 100 % 10) != 0)
    syscall_put_char(num / 100 % 10 + '0');
  if ((num / 10 % 10) != 0)
    syscall_put_char(num / 10 % 10 + '0');
  syscall_put_char(num % 10 + '0');
}

uint8_t load_sector(int16_t segment, int16_t offset, int8_t disc, int8_t sector,
                    uint8_t length) {
  syscall_load_sector(segment, offset, disc, sector, length);
  return syscall_status_last_op(disc);
}

void print_str(const int8_t *str) {
  while (*str != '\0') {
    syscall_put_char(*str);
    str += 1;
  }
}

void syscall_set_cursor_type(uint8_t type, uint8_t mode) {
  asm("mov ah, 0x01 \n"
      "mov ch, %1   \n"
      "mov cl, %0   \n"
      "int 0x10     \n"
      : // no output
      : "g"(type), "g"(mode)
      : "ah", "ch", "cl");
}
void syscall_move_cursor(uint8_t row, uint8_t col) {
  asm("mov ah, 0x02 \n"
      "mov bh, 0x00 \n"
      "mov dh, %0   \n"
      "mov dl, %1   \n"
      "int 0x10     \n"
      : // no output
      : "g"(row), "g"(col)
      : "ah", "bh", "dx");
}

// not yet usable
uint8_t strcmp(int8_t *left, int8_t *right) {
  while (*left != '\0' || *right != '\0') {
    if (*left != *right) {
      return 1;
    }
  }
  return 0;
}
