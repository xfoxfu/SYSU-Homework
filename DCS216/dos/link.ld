OUTPUT_FORMAT(binary)
ENTRY(start)
SECTIONS
{
    . = 0x0100;
    .entry :
    {
        *(.entry);
    }
    .text :
    {
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
