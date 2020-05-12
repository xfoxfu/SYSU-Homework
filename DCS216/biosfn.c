#define VGA_BUFFER ((char *)0xB8000)

void syscall_display_set_char(int row, int col, char chr, unsigned char color) {
  VGA_BUFFER[(row * 80 + col) * 2] = chr;
  VGA_BUFFER[(row * 80 + col) * 2 + 1] = color;
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

void syscall_display_get_char(int row, int col, char *chr, char *color) {
  *chr = VGA_BUFFER[(row * 80 + col) * 2];
  *color = VGA_BUFFER[(row * 80 + col) * 2 + 1];
}
