OUTPUT_FORMAT(binary)
ENTRY(start)
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
        *(.bss);
        *(.rodata);
    }
    _heap = ALIGN(4);
}
