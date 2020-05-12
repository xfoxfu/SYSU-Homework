#define VGA_BUFFER 0xB8000

void syscall_display_set_char(int row, int col, char chr, unsigned char color) {
  ((char *)VGA_BUFFER)[(row * 80 + col) * 2] = chr;
  ((char *)VGA_BUFFER)[(row * 80 + col) * 2 + 1] = color;
}

void syscall_sleep(int time_ms) {
  asm volatile("mov cx, 0x0000 \n"
               "mov ah, 0x86   \n"
               "mov al, 0x00   \n"
               "int 0x15       \n"
               : /* no output */
               : "d"(time_ms)
               : "ah");
}

void syscall_display_get_char(int row, int col, char *chr,
                              unsigned char *color) {
  *chr = ((char *)VGA_BUFFER)[(row * 80 + col) * 2];
  *color = ((char *)VGA_BUFFER)[(row * 80 + col) * 2 + 1];
}

#define VGA_Black 0x0
#define VGA_Blue 0x1
#define VGA_Green 0x2
#define VGA_Cyan 0x3
#define VGA_Red 0x4
#define VGA_Magenta 0x5
#define VGA_Brown 0x6
#define VGA_LightGray 0x7
#define VGA_DarkGray 0x8
#define VGA_LightBlue 0x9
#define VGA_LightGreen 0xa
#define VGA_LightCyan 0xb
#define VGA_LightRed 0xc
#define VGA_Pink 0xd
#define VGA_Yellow 0xe
#define VGA_White 0xf

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

    int col_old = col;
    int row_old = row;

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

    syscall_sleep(25000);

    for (int i = 0; i < 8; i++) {
      if (col_old + 1 + i > 79) {
        break;
      }

      syscall_display_set_char(row_old, col_old + 1 + i, protect[2 * i],
                               protect[2 * i + 1]);
    }
  }
}