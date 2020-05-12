void syscall_display_set_char(int row, int col, char chr, unsigned char color);
void syscall_sleep(int time_ms);
void syscall_display_get_char(int row, int col, char *chr, char *color);
void display(short x, short y, short xm, short ym);

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
