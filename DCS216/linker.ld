OUTPUT_FORMAT(binary)
ENTRY(_start)
SECTIONS
{
    . = 0x0100;
    .text :
    {
        *(.start);
        *(.text);
    }
    .data :
    {
        *(.data);
    }
    .bss :
    {
        *(.bss);
    }
    .rodata :
    {
        *(.rodata);
    }
    _heap = ALIGN(4);
}
