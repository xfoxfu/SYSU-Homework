#include "biosfn.h"

void kmain(void) {
  char NAME[] = {'1', '7', '3', '4', '1', '0', '3', '9'};
  int col = 0;
  int row = 0;
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
      if (col + 1 + i > 79) {
        break;
      }
      syscall_display_set_char(row, col + 1 + i, NAME[i], color);
    }

    syscall_sleep(25000);

    for (int i = 0; i < 8; i++) {
      if (col + 1 + i > 79) {
        break;
      }

      syscall_display_set_char(row, col + 1 + i, protect[2 * i],
                               protect[2 * i + 1]);
    }

    col += col_incr;
    row += row_incr;
    if (col == 0)
      col_incr = 1;
    if (col == 79)
      col_incr = -1;
    if (row == 0)
      row_incr = 1;
    if (row == 23)
      row_incr = -1;

    if (col == 0 || col == 79 || row == 0 || row == 23) {
      color += 1;
    }
    if (color > VGA_White)
      color = VGA_Blue;
  }
}