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
    call dword kmain

    mov  ax,  cs
	mov  ds,  ax

    lea si, [msg_exit]
    call print_str

    retf

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

global syscall_far_jump_A00
syscall_far_jump_A00:
    pusha
    call word 0x0A00:0x0100 ; TODO: find a way of using this in inline assembly
    popa
    mov  ax,  cs
    mov  ds,  ax
    mov  es,  ax
    ret

section .data
msg_ok:   db "Program entry" , 0x0D, 0x0A, 0x00
msg_exit: db "Program exited", 0x0D, 0x0A, 0x00
