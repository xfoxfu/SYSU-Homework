brew tap nativeos/i386-elf-toolchain
brew install i386-elf-binutils i386-elf-gcc

\$ brew install x86_64-elf-gcc x86_64-elf-binutils

\$ x86_64-elf-g++ -m32 kmain.cpp boot4.o -o kernel.bin -nostdlib -ffreestanding -std=c++11 -mno-red-zone -fno-exceptions -nostdlib -fno-rtti -Wall -Wextra -Werror -T linker.ld -Xlinker -melf_i386

\$ nasm -f elf32 boot4.asm -o boot4.o

x86_64-elf-gcc -std=gnu99 -Os -nostdlib -m16 -march=i386 -ffreestanding -o main.com -Wl,--nmagic,--script=link.ld main.c crt0.o -Xlinker -melf_i386
nasm -f elf32 crt0.asm -o crt0.o

需要将入口点定义为特别的段从而保证输出顺序

粗心的忘记定义段和 code 16bits

-Os 似乎有 bug

char call() { return 'a'; } 有返回值就不行

char NAME[] = {'1', '7', '3', '4', '1', '0', '3', '9'}; 全局变量段不行

设置 DS 以后不能用 0xB8000，直接用系统调用
