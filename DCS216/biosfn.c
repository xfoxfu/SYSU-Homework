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

void display(int8_t x, int8_t y, int8_t xm, int8_t ym) {
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
  }
}
