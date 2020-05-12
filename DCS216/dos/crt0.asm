bits 16

section .entry
global start
start:
    mov  ax,  cs
	mov  ds,  ax
	mov  es,  ax
	; mov  ss,  ax
	; mov  sp,  100h
    extern message
    mov dx, message
    mov ah, 0x09
    int 0x21
    ; push message
    extern dosmain
    call  dosmain
    add   sp, 0x2
    mov   dx, message
    mov   ah, 0x09
    int   0x21
    mov   al, 0x00
    mov   ah, 0x4C
    int   0x21

section .data
hello: db "acbdfewajk$"

section .text
