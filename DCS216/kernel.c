#define VGA_BUFFER 0xB8000

void sleep(int time) {
  asm volatile("mov $0x0000, %%cx\n"
               "mov $0x86,   %%ah\n"
               "mov $0x00,   %%al\n"
               "int $0x15\n"
               : /* no output */
               : "d"(time)
               : "ah");
}

void kmain(void) {
  char *table = (char *)VGA_BUFFER;
  table[1] = 'X';

  int col = 0;
  int row = 0;
  int col_incr = 1;
  int row_incr = 1;
  for (;;) {
    int pos = (row * 80 + col) * 2;
    ((char *)table)[pos + 0 * 2] = '1';
    ((char *)table)[pos + 1 * 2] = '7';
    ((char *)table)[pos + 2 * 2] = '3';
    ((char *)table)[pos + 3 * 2] = '4';
    ((char *)table)[pos + 4 * 2] = '1';
    ((char *)table)[pos + 5 * 2] = '0';
    ((char *)table)[pos + 6 * 2] = '3';
    ((char *)table)[pos + 7 * 2] = '9';

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

    sleep(0xC350);
  }
}