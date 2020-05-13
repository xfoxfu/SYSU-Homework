; /usr/local/bin/nasm

bits 16
section .start
global _start
_start:
    mov  ax,  cs
	mov  ds,  ax
	mov  es,  ax
    mov  fs,  ax

    mov word [p_ss], ss ; move stack to a local area will effectively fix some bug
    mov word [p_sp], sp
    mov word [p_bp], bp
    mov  ss,  ax
    mov  sp,  100h
    mov  bp,  100h

    extern kmain
    call dword kmain

    mov ss, word [p_ss]
    mov sp, word [p_sp]
    mov bp, word [p_bp]

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

section .data
msg_ok:   db "Program entry" , 0x0D, 0x0A, 0x00
msg_exit: db "Program exited", 0x0D, 0x0A, 0x00
p_ss:     db 0x00, 0x00
p_sp:     db 0x00, 0x00
p_bp:     db 0x00, 0x00
