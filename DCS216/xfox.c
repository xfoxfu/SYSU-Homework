#include <stdio.h>
#include <unistd.h>

int main() {
  char table[24][80]; // char is better

  memset(table, ' ', sizeof(table[0][0]) * 24 * 80); // Fill table with spaces

  int col = 0;
  int row = 0;
  int col_incr = 1;
  int row_incr = 1;
  for (;;) {
    int pos = row * 80 + col;
    ((char *)table)[pos + 0] = '1';
    ((char *)table)[pos + 1] = '7';
    ((char *)table)[pos + 2] = '3';
    ((char *)table)[pos + 3] = '4';
    ((char *)table)[pos + 4] = '1';
    ((char *)table)[pos + 5] = '0';
    ((char *)table)[pos + 6] = '3';
    ((char *)table)[pos + 7] = '9';

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

    printf("\033[2J\033[%d;%d0H", 0, 0);
    printf("==================================================================="
           "=========\n");
    for (int j = 0; j < 24; ++j) {
      for (int i = 0; i < 80; ++i) {
        printf("%c", table[j][i]);
      }
      printf("\n");
    }
    usleep(1000 * 1000 * 0.1);
  }
  return 0;
}
