; /usr/local/bin/nasm

bits 16
section .start
global _start
_start:
    mov  ax,  cs
	mov  ds,  ax
	mov  es,  ax

    lea si, [msg_ok]
    call print_str

    xor ax, ax
    mov ds, ax

    extern kmain
    call kmain

    lea si, [msg_exit]
    call print_str
    jmp $

print_str: ; IN si start address
    lodsb
    cmp al, 0
    je .done
    mov ah, 0xE
    mov bx, 7
    int 10h
    jmp print_str
    .done:
        ret

section .data
msg_ok:   db "Kernel entry" , 0x0D, 0x0A, 0x00
msg_exit: db "Kernel exited", 0x0D, 0x0A, 0x00
