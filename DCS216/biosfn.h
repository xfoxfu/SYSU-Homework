#include <stdint.h>

void syscall_display_set_char(int16_t row, int16_t col, int8_t chr,
                              uint8_t color);
void syscall_sleep(int16_t time_ms);
void syscall_display_get_char(int16_t row, int16_t col, int8_t *chr,
                              int8_t *color);
void display(int8_t x, int8_t y, int8_t xm, int8_t ym);
int8_t syscall_get_key_noblock(void);

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
