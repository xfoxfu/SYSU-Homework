#!/bin/sh
nasm -f elf32 crt0.asm -o crt0.o
x86_64-elf-gcc -std=gnu99 -Os -nostdlib -m16 -march=i386 -ffreestanding -o main.com -Wl,--nmagic,--script=link.ld main.c crt0.o -Xlinker -melf_i386 -estart
qemu-system-x86_64 ../freedos.img -drive file=fat:rw:./