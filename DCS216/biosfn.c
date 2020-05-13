#include "biosfn.h"

#define VGA_BUFFER ((char *)0xB8000)

void syscall_display_set_char(int16_t row, int16_t col, int8_t chr,
                              uint8_t color) {
  VGA_BUFFER[(row * 80 + col) * 2] = chr;
  VGA_BUFFER[(row * 80 + col) * 2 + 1] = color;
}

void syscall_sleep(int16_t time_ms) {
  asm volatile("mov cx, 0x0000 \n"
               "mov ah, 0x86   \n"
               "mov al, 0x00   \n"
               "int 0x15       \n"
               : /* no output */
               : "d"(time_ms)
               : "ah");
}

void syscall_display_get_char(int16_t row, int16_t col, int8_t *chr,
                              int8_t *color) {
  *chr = VGA_BUFFER[(row * 80 + col) * 2];
  *color = VGA_BUFFER[(row * 80 + col) * 2 + 1];
}

void display(const int8_t x, const int8_t y, const int8_t xm, const int8_t ym) {
  char NAME[] = {'1', '7', '3', '4', '1', '0', '3', '9'};
  int col = x;
  int row = y;
  int col_incr = 1;
  int row_incr = 1;
  int color = VGA_White;
  char protect[8 * 2] = {};

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
      : "ax");
  return key;
}

int8_t syscall_get_key_block(void) {
  int8_t key;
  asm("mov ah, 0x00 \n"
      "int 0x16     \n"
      "mov %0, al   \n"
      : "=r"(key)
      : /* no input */
      : "ax");
  return key;
}

int8_t syscall_get_default_drive(void) {
  int8_t drive;
  asm("mov ah, 0x0E \n"
      "int 0x21     \n"
      "mov %0, dl   \n"
      : "=r"(drive)
      : /* no input */
      : "al");
  return drive;
}

int8_t syscall_status_last_op(int8_t disc) {
  int8_t status;
  asm("mov ah, 0x01 \n"
      "mov dl, %0   \n"
      "int 0x13     \n"
      "mov %1, ah   \n"
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
      : "bx");
}

void syscall_load_sector(int16_t segment, int16_t offset, int8_t disc,
                         int8_t sector) {
  asm volatile("push es        \n"
               "mov  bx, %0     \n"
               "mov  es, bx     \n"
               "mov  bx, %1     \n"
               "mov  ax, 0x0201 \n"
               "mov  dh, 0x00   \n"
               "mov  dl, %2     \n"
               "mov  ch, 0x00   \n"
               "mov  cl, %3     \n"
               "int  0x13       \n"
               "pop  es         \n"
               : /* no output */
               : "g"(segment), "g"(offset), "g"(disc), "g"(sector)
               // cannot put these as registers
               : "ah", "al", "memory");
}
