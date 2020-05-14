#include "biosfn.h"
#include <stdint.h>

void kmain(void) {
  syscall_display_mode(0x13);

  int x = 0;
  int y = 0;
  int xm = 320;
  int ym = 200;

  for (int i = x; i < xm; i++) {
    for (int j = y; j < ym; j++) {
      raw_far_pointer_write(0xA000, i + j * 320, VGA_Blue);
    }
  }

  int col = x;
  int row = y;
  int col_incr = 1;
  int row_incr = 1;
  int color = VGA_White;
  char protect[8 * 2] = {0};

  for (;;) {
    raw_far_pointer_write(0xA000, col + row * 320, color);

    syscall_sleep(5000);

    col += col_incr;
    row += row_incr;
    if (col <= x)
      col_incr = 1;
    if (col >= xm - 1)
      col_incr = -1;
    if (row <= y)
      row_incr = 1;
    if (row >= ym - 1)
      row_incr = -1;

    if (col == x || col == xm - 1 || row == y || row == ym - 1) {
      color += 1;
    }
    if (color > VGA_White)
      color = VGA_Cyan;
    if (color == VGA_Blue)
      color += 1;

    int8_t ch = syscall_get_key_noblock();
    if (ch == 0x1B) {
      syscall_display_mode(0x03);
      return;
    }
  }
}
